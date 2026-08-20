#include "TamHook.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

namespace rcs {
namespace hw {

namespace {

// Default per-joint clip of the TAM residual (Nm).
constexpr double kDefaultAdaptorTorqueLimitsNm[7] = {10.0, 10.0, 10.0, 10.0,
                                                     8.0,  8.0,  8.0};

bool is_effectively_zero_torque(const TamHook::Vec7& tau) {
  return tau.cwiseAbs().maxCoeff() <= TamHook::kZeroTorqueThresholdNm;
}

}  // namespace

TamHook::TamHook(size_t max_history) : max_history_(std::max<size_t>(max_history, 16)) {
  for (int i = 0; i < 7; ++i) {
    torque_limits_(i) = kDefaultAdaptorTorqueLimitsNm[i];
  }
  const auto now = std::chrono::steady_clock::now();
  enabled_since_ = now;
  start_time_ = now;
}

// ---------------------------------------------------------------------------
// non-realtime API
// ---------------------------------------------------------------------------

bool TamHook::load_adaptor(const std::string& weight_path) {
  auto model = adaptor::SimAdaptor::LoadFromFile(weight_path);
  if (!model) {
    std::cerr << "[TamHook] Failed to load SimAdaptor from " << weight_path
              << ". Rebuild the extension if the bin was exported by a newer "
                 "Sim2realAdaptor checkpoint.\n";
    return false;
  }
  std::shared_ptr<adaptor::SimAdaptor> shared(std::move(model));
  std::lock_guard<std::mutex> lock(mux_);
  adaptor_ = std::move(shared);
  adaptor_path_ = weight_path;
  if (enabled_) {
    enabled_since_ = std::chrono::steady_clock::now();
  }
  std::cout << "[TamHook] Loaded SimAdaptor dof=" << adaptor_->dof
            << " emb_dim=" << adaptor_->emb_dim << " hidden=" << adaptor_->hidden
            << " depth=" << adaptor_->depth
            << " history_steps=" << adaptor_->history_steps
            << " flags=" << adaptor_->flags << " from " << weight_path << "\n";
  return true;
}

void TamHook::set_adaptor_for_test(std::shared_ptr<adaptor::SimAdaptor> adaptor,
                                   const std::string& label) {
  std::lock_guard<std::mutex> lock(mux_);
  adaptor_ = std::move(adaptor);
  adaptor_path_ = label;
}

void TamHook::set_embedding(const Eigen::VectorXd& embedding) {
  std::lock_guard<std::mutex> lock(mux_);
  embedding_ = embedding;
  ++embedding_seq_;
}

uint64_t TamHook::embedding_seq() const {
  std::lock_guard<std::mutex> lock(mux_);
  return embedding_seq_;
}

void TamHook::enable(bool enabled) {
  std::lock_guard<std::mutex> lock(mux_);
  const bool was_enabled = enabled_;
  enabled_ = enabled;
  if (enabled && !was_enabled) {
    enabled_since_ = std::chrono::steady_clock::now();
  }
}

bool TamHook::enabled() const {
  std::lock_guard<std::mutex> lock(mux_);
  return enabled_;
}

void TamHook::set_ideal_model_has_gravity(bool enabled) {
  std::lock_guard<std::mutex> lock(mux_);
  ideal_model_has_gravity_ = enabled;
}

bool TamHook::ideal_model_has_gravity() const {
  std::lock_guard<std::mutex> lock(mux_);
  return ideal_model_has_gravity_;
}

void TamHook::set_torque_limits(const Vec7& limits) {
  std::lock_guard<std::mutex> lock(mux_);
  torque_limits_ = limits.cwiseAbs();
}

void TamHook::set_enable_ramp_s(double seconds) {
  std::lock_guard<std::mutex> lock(mux_);
  enable_ramp_s_ = std::max(0.0, seconds);
}

std::vector<TamHook::HistoryRow> TamHook::get_history(size_t max_rows) const {
  std::lock_guard<std::mutex> lock(mux_);
  std::vector<HistoryRow> out;
  out.reserve(std::min(max_rows, history_.size()));
  for (auto it = history_.rbegin(); it != history_.rend() && out.size() < max_rows; ++it) {
    if (!it->publish_ready) {
      continue;
    }
    out.push_back(*it);
  }
  std::reverse(out.begin(), out.end());
  return out;
}

TamHook::Status TamHook::status() const {
  std::lock_guard<std::mutex> lock(mux_);
  Status s;
  s.loaded = static_cast<bool>(adaptor_);
  s.enabled = enabled_;
  s.ideal_model_has_gravity = ideal_model_has_gravity_;
  s.embedding_seq = embedding_seq_;
  s.embedding_size = static_cast<int>(embedding_.size());
  s.expected_embedding_size = adaptor_ ? adaptor_->expected_history_embedding_cols() : 0;
  s.history_steps = adaptor_ ? adaptor_->history_steps : 0;
  s.dof = adaptor_ ? adaptor_->dof : 7;
  s.history_size = history_.size();
  s.rows_total = rows_total_;
  s.start_count = start_count_;
  s.adaptor_forward_dt_ms = adaptor_forward_dt_ms_;
  s.enable_scale = last_enable_scale_;
  s.last_skip_reason = last_skip_reason_;
  s.adaptor_path = adaptor_path_;
  s.torque_limits = torque_limits_;
  return s;
}

// ---------------------------------------------------------------------------
// realtime API
// ---------------------------------------------------------------------------

double TamHook::enable_scale_locked(const std::chrono::steady_clock::time_point& now) const {
  if (!enabled_) {
    return 0.0;
  }
  const double enabled_for = std::chrono::duration<double>(now - enabled_since_).count();
  if (enabled_for <= 0.0) {
    return 0.0;
  }
  if (enable_ramp_s_ <= 0.0 || enabled_for >= enable_ramp_s_) {
    return 1.0;
  }
  return enabled_for / enable_ramp_s_;
}

void TamHook::on_control_start() {
  std::lock_guard<std::mutex> lock(mux_);
  const auto now = std::chrono::steady_clock::now();
  history_.clear();
  embedding_.resize(0);
  embedding_seq_ = 0;
  start_time_ = now;
  started_ = true;
  ++start_count_;
  if (enabled_) {
    enabled_since_ = now;
  }
  pending_delta_.setZero();
  pending_adaptor_used_ = false;
  last_skip_reason_ = "no_rows";
}

TamHook::Vec7 TamHook::apply(double period_sec, const Vec7& q, const Vec7& dq,
                             const Vec7& tau_base, const Vec7& gravity,
                             const Vec7& tau_commanded, const Vec7& tau_measured) {
  Vec7 delta = Vec7::Zero();
  const auto now = std::chrono::steady_clock::now();

  adaptor::M hist_m;
  adaptor::M q_hist;
  adaptor::M dq_hist;
  adaptor::M tau_hist;
  std::shared_ptr<adaptor::SimAdaptor> adaptor_ptr;
  double enable_scale = 0.0;
  bool ideal_model_has_gravity = true;
  Vec7 torque_limits = Vec7::Zero();
  std::string skip_reason;

  {
    std::lock_guard<std::mutex> lock(mux_);
    if (!started_) {
      start_time_ = now;
      started_ = true;
      ++start_count_;
    }
    const double sample_t_sec = std::chrono::duration<double>(now - start_time_).count();
    const double step_dt_sec = period_sec > 0.0 ? period_sec : kHistorySampleDtSec;
    const int step_slots =
        std::max(1, static_cast<int>(std::llround(step_dt_sec / kHistorySampleDtSec)));
    const int missing_slots = std::min(std::max(0, step_slots - 1), kMaxPaddingRowsPerStep);
    const bool current_valid_for_history = !is_effectively_zero_torque(tau_base);
    const uint64_t history_embedding_seq = embedding_seq_;

    auto append_row = [&](HistoryRow row) {
      history_.push_back(std::move(row));
      if (history_.size() > max_history_) {
        history_.pop_front();
      }
      ++rows_total_;
    };

    for (int slot = missing_slots; slot > 0; --slot) {
      HistoryRow padding;
      padding.t = std::max(0.0, sample_t_sec - slot * kHistorySampleDtSec);
      padding.history_embedding_seq = history_embedding_seq;
      padding.valid_for_history = false;
      padding.synthetic_padding = true;
      padding.publish_ready = true;
      padding.sample_dt_sec = kHistorySampleDtSec;
      append_row(std::move(padding));
    }

    HistoryRow row;
    row.t = sample_t_sec;
    row.q = q;
    row.dq = dq;
    row.tau_base = tau_base;
    row.tau_commanded = tau_commanded;
    row.tau_measured = tau_measured;
    row.gravity = gravity;
    row.history_embedding_seq = history_embedding_seq;
    row.valid_for_history = current_valid_for_history;
    row.synthetic_padding = false;
    row.publish_ready = false;  // finalize_row() flips this
    row.sample_dt_sec = step_dt_sec;
    append_row(std::move(row));

    enable_scale = enable_scale_locked(now);
    last_enable_scale_ = enable_scale;
    ideal_model_has_gravity = ideal_model_has_gravity_;
    torque_limits = torque_limits_;

    const bool have_adaptor = enabled_ && adaptor_;
    const int expected_emb_cols = have_adaptor ? adaptor_->expected_history_embedding_cols() : 0;
    const bool emb_ok =
        have_adaptor && embedding_.size() > 0 && embedding_.size() == expected_emb_cols;
    const bool history_ok =
        have_adaptor && history_.size() >= static_cast<size_t>(adaptor_->history_steps);

    if (have_adaptor && emb_ok && history_ok) {
      adaptor_ptr = adaptor_;
      hist_m.resize(1, expected_emb_cols);
      for (int i = 0; i < expected_emb_cols; ++i) {
        hist_m(0, i) = static_cast<float>(embedding_(i));
      }
      const int T = adaptor_->history_steps;
      const int D = adaptor_->dof;
      q_hist.resize(1, T * D);
      dq_hist.resize(1, T * D);
      tau_hist.resize(1, T * D);
      auto it = history_.end();
      for (int t = 0; t < T; ++t) {
        --it;
        const HistoryRow& s = *it;
        const int offset = (T - 1 - t) * D;  // earliest history at lowest offset
        const bool row_valid = (t == 0) ? current_valid_for_history : s.valid_for_history;
        Vec7 tau_src = Vec7::Zero();
        if (row_valid) {
          tau_src = (t == 0) ? tau_base : s.tau_applied;
          if (ideal_model_has_gravity) {
            tau_src += (t == 0) ? gravity : s.gravity;
          }
        }
        for (int j = 0; j < D && j < 7; ++j) {
          q_hist(0, offset + j) = row_valid ? static_cast<float>(s.q(j)) : 0.0f;
          dq_hist(0, offset + j) = row_valid ? static_cast<float>(s.dq(j)) : 0.0f;
          tau_hist(0, offset + j) = static_cast<float>(tau_src(j));
        }
      }
      last_q_hist_ = q_hist;
      last_dq_hist_ = dq_hist;
      last_tau_hist_ = tau_hist;
      last_emb_row_ = hist_m;
    } else if (!enabled_) {
      skip_reason = "disabled";
    } else if (!adaptor_) {
      skip_reason = "not_loaded";
    } else if (!emb_ok) {
      if (embedding_.size() == 0) {
        skip_reason = "missing_embedding";
      } else {
        skip_reason = "embedding_dim_mismatch";
      }
    } else if (!history_ok) {
      skip_reason = "insufficient_history";
    } else {
      skip_reason = "unknown";
    }
  }

  bool adaptor_used = false;
  double forward_ms = 0.0;
  if (adaptor_ptr) {
    const auto t0 = std::chrono::steady_clock::now();
    adaptor::M delta_tau;
    try {
      delta_tau = adaptor_ptr->forward(q_hist, dq_hist, tau_hist, hist_m);
    } catch (...) {
      delta_tau.resize(0, 0);
      skip_reason = "forward_exception";
    }
    const auto t1 = std::chrono::steady_clock::now();
    forward_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (delta_tau.size() == adaptor_ptr->dof && adaptor_ptr->dof <= 7) {
      for (int i = 0; i < adaptor_ptr->dof; ++i) {
        const double v = static_cast<double>(delta_tau(0, i));
        const double lim = torque_limits(i);
        delta(i) = std::isfinite(v) ? std::clamp(v, -lim, lim) : 0.0;
      }
      delta *= enable_scale;
      adaptor_used = true;
    } else if (skip_reason.empty()) {
      skip_reason = "delta_size_" + std::to_string(delta_tau.size());
    }
  }

  {
    std::lock_guard<std::mutex> lock(mux_);
    adaptor_forward_dt_ms_ = forward_ms;
    pending_delta_ = delta;
    pending_adaptor_used_ = adaptor_used;
    if (!skip_reason.empty()) {
      last_skip_reason_ = skip_reason;
    } else if (adaptor_used) {
      last_skip_reason_.clear();
    }
  }
  return delta;
}

void TamHook::finalize_row(const Vec7& tau_applied) {
  std::lock_guard<std::mutex> lock(mux_);
  if (history_.empty()) {
    return;
  }
  HistoryRow& row = history_.back();
  if (row.publish_ready) {
    return;  // nothing pending (apply() was not called this tick)
  }
  row.tau_applied = tau_applied;
  row.tau_adaptor_delta = pending_delta_;
  row.adaptor_active = pending_adaptor_used_;
  row.valid_for_history = !is_effectively_zero_torque(tau_applied);
  // Set last while holding mux_: get_history() filters unfinished rows.
  row.publish_ready = true;
  pending_delta_.setZero();
  pending_adaptor_used_ = false;
}

adaptor::M TamHook::last_q_hist() const {
  std::lock_guard<std::mutex> lock(mux_);
  return last_q_hist_;
}

adaptor::M TamHook::last_dq_hist() const {
  std::lock_guard<std::mutex> lock(mux_);
  return last_dq_hist_;
}

adaptor::M TamHook::last_tau_hist() const {
  std::lock_guard<std::mutex> lock(mux_);
  return last_tau_hist_;
}

adaptor::M TamHook::last_embedding_row() const {
  std::lock_guard<std::mutex> lock(mux_);
  return last_emb_row_;
}

}  // namespace hw
}  // namespace rcs
