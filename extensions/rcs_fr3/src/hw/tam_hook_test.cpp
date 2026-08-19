// Standalone unit test for TamHook (Eigen only, no libfranka).
//
// Build (from this directory):
//   c++ -std=c++17 -O1 -I. -I<eigen3 include dir> tam_hook_test.cpp TamHook.cpp -o tam_hook_test
// or configure the extension with -DRCS_FR3_BUILD_TAM_HOOK_TEST=ON.

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "TamHook.h"

namespace {

int g_failures = 0;

#define EXPECT_TRUE(cond)                                                            \
  do {                                                                               \
    if (!(cond)) {                                                                   \
      std::cerr << __FILE__ << ":" << __LINE__ << ": EXPECT_TRUE failed: " #cond "\n"; \
      ++g_failures;                                                                  \
    }                                                                                \
  } while (0)

#define EXPECT_NEAR(a, b, tol)                                                          \
  do {                                                                                  \
    const double _a = (a);                                                              \
    const double _b = (b);                                                              \
    if (!(std::fabs(_a - _b) <= (tol))) {                                               \
      std::cerr << __FILE__ << ":" << __LINE__ << ": EXPECT_NEAR failed: " << _a         \
                << " vs " << _b << " (tol " << (tol) << ")\n";                          \
      ++g_failures;                                                                     \
    }                                                                                   \
  } while (0)

using Vec7 = rcs::hw::TamHook::Vec7;

void init_dense(adaptor::Dense& d, float w, float b) {
  d.W.setConstant(w);
  d.b.setConstant(b);
}

void init_ln(adaptor::LayerNorm& ln) {
  ln.gamma.setOnes();
  ln.beta.setZero();
}

void init_block(adaptor::SimAdaptorBlock& blk, float out_bias) {
  init_ln(blk.adaln1.ln);
  init_dense(blk.adaln1.proj, 0.0f, 0.0f);
  init_dense(blk.fc1, 0.0f, 0.0f);
  init_ln(blk.adaln2.ln);
  init_dense(blk.adaln2.proj, 0.0f, 0.0f);
  init_dense(blk.fc2, 0.0f, out_bias);
}

// Global (non-jointwise) adaptor whose output equals a constant bias.
std::shared_ptr<adaptor::SimAdaptor> make_constant_adaptor(int dof, int emb_dim, int hidden,
                                                           int depth, int history_steps,
                                                           float out_value) {
  auto m = std::make_shared<adaptor::SimAdaptor>(dof, emb_dim, hidden, depth, history_steps);
  init_dense(m->q_stem, 0.0f, 0.0f);
  init_dense(m->qd_stem, 0.0f, 0.0f);
  init_dense(m->tau_stem, 0.0f, 0.0f);
  init_ln(m->ln_q);
  init_ln(m->ln_qd);
  init_ln(m->ln_tau);
  for (auto& blk : m->blocks) {
    init_block(blk, 0.0f);
  }
  init_block(m->out_block, out_value);
  init_block(m->external_torque_block, 0.0f);
  m->norm_stats.disable();
  return m;
}

Vec7 vec7(double v) { return Vec7::Constant(v); }

void test_no_adaptor_paths() {
  rcs::hw::TamHook hook(64);
  hook.on_control_start();
  Vec7 delta = hook.apply(0.001, vec7(0.1), vec7(0.0), vec7(1.0), vec7(2.0), vec7(0.0), vec7(0.0));
  EXPECT_TRUE(delta.isZero());
  EXPECT_TRUE(hook.status().last_skip_reason == "disabled");
  hook.finalize_row(vec7(1.0));
  hook.enable(true);
  delta = hook.apply(0.001, vec7(0.1), vec7(0.0), vec7(1.0), vec7(2.0), vec7(0.0), vec7(0.0));
  EXPECT_TRUE(delta.isZero());
  EXPECT_TRUE(hook.status().last_skip_reason == "not_loaded");
  hook.finalize_row(vec7(1.0));
  EXPECT_TRUE(hook.get_history(10).size() == 2);
}

void test_publish_ready_and_padding() {
  rcs::hw::TamHook hook(64);
  hook.on_control_start();
  hook.apply(0.001, vec7(0.1), vec7(0.2), vec7(1.0), vec7(2.0), vec7(0.5), vec7(0.6));
  // The row appended by apply() is not visible until finalize_row().
  EXPECT_TRUE(hook.get_history(10).empty());
  hook.finalize_row(vec7(1.5));
  auto rows = hook.get_history(10);
  EXPECT_TRUE(rows.size() == 1);
  if (!rows.empty()) {
    EXPECT_NEAR(rows[0].q(0), 0.1, 1e-12);
    EXPECT_NEAR(rows[0].dq(0), 0.2, 1e-12);
    EXPECT_NEAR(rows[0].tau_base(0), 1.0, 1e-12);
    EXPECT_NEAR(rows[0].gravity(0), 2.0, 1e-12);
    EXPECT_NEAR(rows[0].tau_commanded(0), 0.5, 1e-12);
    EXPECT_NEAR(rows[0].tau_measured(0), 0.6, 1e-12);
    EXPECT_NEAR(rows[0].tau_applied(0), 1.5, 1e-12);
    EXPECT_TRUE(rows[0].publish_ready);
    EXPECT_TRUE(rows[0].valid_for_history);
    EXPECT_TRUE(!rows[0].synthetic_padding);
    EXPECT_TRUE(!rows[0].adaptor_active);
  }
  // A 3 ms period inserts two synthetic padding rows before the real row
  // (sleep so the padded timestamps are not clamped at the time origin).
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  hook.apply(0.003, vec7(0.1), vec7(0.2), vec7(1.0), vec7(2.0), vec7(0.5), vec7(0.6));
  hook.finalize_row(vec7(1.0));
  rows = hook.get_history(10);
  EXPECT_TRUE(rows.size() == 4);
  if (rows.size() == 4) {
    EXPECT_TRUE(rows[1].synthetic_padding && !rows[1].valid_for_history && rows[1].publish_ready);
    EXPECT_TRUE(rows[2].synthetic_padding);
    EXPECT_TRUE(!rows[3].synthetic_padding);
    EXPECT_TRUE(rows[1].t < rows[2].t && rows[2].t < rows[3].t);
  }
  // Zero applied torque marks the row invalid for the history encoder.
  hook.apply(0.001, vec7(0.1), vec7(0.2), vec7(0.0), vec7(2.0), vec7(0.5), vec7(0.6));
  hook.finalize_row(vec7(0.0));
  rows = hook.get_history(10);
  EXPECT_TRUE(!rows.back().valid_for_history);
  // Ring buffer bound.
  for (int i = 0; i < 200; ++i) {
    hook.apply(0.001, vec7(0.1), vec7(0.2), vec7(1.0), vec7(2.0), vec7(0.5), vec7(0.6));
    hook.finalize_row(vec7(1.0));
  }
  EXPECT_TRUE(hook.status().history_size <= 64);
  EXPECT_TRUE(hook.get_history(1000).size() <= 64);
}

void test_adaptor_gating_clip_and_gravity() {
  const int dof = 7, emb_dim = 4, hidden = 8, depth = 2, history_steps = 3;
  rcs::hw::TamHook hook(64);
  hook.set_adaptor_for_test(make_constant_adaptor(dof, emb_dim, hidden, depth, history_steps, 100.0f));
  hook.set_enable_ramp_s(0.0);
  hook.on_control_start();
  hook.enable(true);

  // Missing embedding -> skipped.
  Vec7 delta = hook.apply(0.001, vec7(0.1), vec7(0.0), vec7(1.0), vec7(2.0), vec7(0.0), vec7(0.0));
  hook.finalize_row(vec7(1.0));
  EXPECT_TRUE(delta.isZero());
  EXPECT_TRUE(hook.status().last_skip_reason == "missing_embedding");

  // Wrong embedding size -> skipped.
  hook.set_embedding(Eigen::VectorXd::Ones(emb_dim + 1));
  delta = hook.apply(0.001, vec7(0.1), vec7(0.0), vec7(1.0), vec7(2.0), vec7(0.0), vec7(0.0));
  hook.finalize_row(vec7(1.0));
  EXPECT_TRUE(delta.isZero());
  EXPECT_TRUE(hook.status().last_skip_reason == "embedding_dim_mismatch");
  EXPECT_TRUE(hook.embedding_seq() == 1);

  // Correct embedding: history has 2 rows so far -> after this apply() there
  // are 3 rows == history_steps -> forward runs, output 100 clipped per joint.
  hook.set_embedding(Eigen::VectorXd::Ones(emb_dim));
  Vec7 tau_base = vec7(1.0);
  tau_base(6) = 3.0;
  Vec7 gravity = vec7(2.0);
  gravity(6) = -1.0;
  delta = hook.apply(0.001, vec7(0.1), vec7(0.0), tau_base, gravity, vec7(0.0), vec7(0.0));
  EXPECT_TRUE(hook.status().last_skip_reason.empty());
  EXPECT_NEAR(delta(0), 10.0, 1e-9);
  EXPECT_NEAR(delta(4), 8.0, 1e-9);
  EXPECT_NEAR(delta(6), 8.0, 1e-9);
  hook.finalize_row(tau_base + delta);
  auto rows = hook.get_history(10);
  EXPECT_TRUE(rows.back().adaptor_active);
  EXPECT_NEAR(rows.back().tau_adaptor_delta(0), 10.0, 1e-9);
  EXPECT_NEAR(rows.back().tau_applied(6), 11.0, 1e-9);
  EXPECT_TRUE(rows.back().history_embedding_seq == 2);

  // Adaptor inputs: newest slot uses tau_base (+gravity), older slots tau_applied (+gravity).
  adaptor::M tau_hist = hook.last_tau_hist();
  EXPECT_TRUE(tau_hist.cols() == history_steps * dof);
  if (tau_hist.cols() == history_steps * dof) {
    const int newest = (history_steps - 1) * dof;
    EXPECT_NEAR(tau_hist(0, newest + 0), 1.0 + 2.0, 1e-6);
    EXPECT_NEAR(tau_hist(0, newest + 6), 3.0 - 1.0, 1e-6);
    // previous row: tau_applied = 1.0 (finalized above), gravity 2.0
    EXPECT_NEAR(tau_hist(0, (history_steps - 2) * dof + 0), 1.0 + 2.0, 1e-6);
  }
  adaptor::M q_hist = hook.last_q_hist();
  EXPECT_NEAR(q_hist(0, 0), 0.1, 1e-6);

  // Without ideal-model gravity the torque history is the raw libfranka command.
  hook.set_ideal_model_has_gravity(false);
  delta = hook.apply(0.001, vec7(0.1), vec7(0.0), tau_base, gravity, vec7(0.0), vec7(0.0));
  hook.finalize_row(tau_base + delta);
  tau_hist = hook.last_tau_hist();
  EXPECT_NEAR(tau_hist(0, (history_steps - 1) * dof + 0), 1.0, 1e-6);
  EXPECT_NEAR(tau_hist(0, (history_steps - 1) * dof + 6), 3.0, 1e-6);

  // Custom clip limits are honoured.
  hook.set_torque_limits(vec7(2.5));
  delta = hook.apply(0.001, vec7(0.1), vec7(0.0), tau_base, gravity, vec7(0.0), vec7(0.0));
  hook.finalize_row(tau_base + delta);
  EXPECT_NEAR(delta(0), 2.5, 1e-9);

  // Disable -> zero delta, rows keep flowing.
  hook.enable(false);
  delta = hook.apply(0.001, vec7(0.1), vec7(0.0), tau_base, gravity, vec7(0.0), vec7(0.0));
  hook.finalize_row(tau_base);
  EXPECT_TRUE(delta.isZero());
  EXPECT_TRUE(!hook.get_history(1).back().adaptor_active);

  // Enable ramp: right after enabling with a 1 s ramp the scale is small.
  hook.set_enable_ramp_s(1.0);
  hook.enable(true);
  delta = hook.apply(0.001, vec7(0.1), vec7(0.0), tau_base, gravity, vec7(0.0), vec7(0.0));
  hook.finalize_row(tau_base + delta);
  EXPECT_TRUE(delta(0) >= 0.0 && delta(0) < 2.5 * 0.5);
  auto st = hook.status();
  EXPECT_TRUE(st.loaded && st.enabled);
  EXPECT_TRUE(st.expected_embedding_size == emb_dim);
  EXPECT_TRUE(st.history_steps == history_steps);

  // on_control_start clears rows and the embedding.
  hook.on_control_start();
  EXPECT_TRUE(hook.get_history(10).empty());
  EXPECT_TRUE(hook.embedding_seq() == 0);
  delta = hook.apply(0.001, vec7(0.1), vec7(0.0), tau_base, gravity, vec7(0.0), vec7(0.0));
  hook.finalize_row(tau_base);
  EXPECT_TRUE(hook.status().last_skip_reason == "missing_embedding");
  EXPECT_TRUE(hook.status().start_count == 2);
}

void test_reader_thread_concurrency() {
  const int dof = 7, emb_dim = 4, hidden = 8, depth = 2, history_steps = 5;
  rcs::hw::TamHook hook(256);
  hook.set_adaptor_for_test(make_constant_adaptor(dof, emb_dim, hidden, depth, history_steps, 0.5f));
  hook.set_enable_ramp_s(0.0);
  hook.on_control_start();
  hook.enable(true);
  hook.set_embedding(Eigen::VectorXd::Ones(emb_dim));
  std::atomic<bool> stop{false};
  std::atomic<long> reads{0};
  std::thread reader([&]() {
    while (!stop.load()) {
      auto rows = hook.get_history(50);
      for (const auto& r : rows) {
        if (!r.publish_ready) {
          ++g_failures;
        }
      }
      hook.set_embedding(Eigen::VectorXd::Constant(emb_dim, 0.5));
      ++reads;
    }
  });
  for (int i = 0; i < 5000; ++i) {
    Vec7 delta = hook.apply(0.001, vec7(0.1), vec7(0.0), vec7(1.0), vec7(2.0), vec7(0.0), vec7(0.0));
    hook.finalize_row(vec7(1.0) + delta);
  }
  stop.store(true);
  reader.join();
  EXPECT_TRUE(reads.load() > 0);
  EXPECT_TRUE(hook.status().rows_total == 5000);
}

}  // namespace

int main() {
  test_no_adaptor_paths();
  test_publish_ready_and_padding();
  test_adaptor_gating_clip_and_gravity();
  test_reader_thread_concurrency();
  if (g_failures == 0) {
    std::cout << "tam_hook_test: all checks passed\n";
    return EXIT_SUCCESS;
  }
  std::cerr << "tam_hook_test: " << g_failures << " check(s) failed\n";
  return EXIT_FAILURE;
}
