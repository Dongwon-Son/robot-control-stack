#ifndef RCS_TAM_HOOK_H
#define RCS_TAM_HOOK_H

// TAM (Torque Adaptation Module) hook for the hw::Franka torque controllers.
//
// The hook sits between the RCS torque law (joint PD / OSC) and the RCS
// rate-limit + torque_limit tail inside the 1 kHz libfranka callback:
//
//   tau_base = <RCS law>;                       // gravity-free (libfranka adds gravity)
//   tau_d    = tau_base + tam.apply(...);       // TAM residual (this class)
//   tau_d    = limitRate(...); clamp(...);      // unchanged RCS tail
//   tam.finalize_row(tau_d);                    // publish the finished history row
//
// It keeps a bounded 1 kHz history ring buffer (q, dq, torque decomposition,
// gravity) that a non-realtime thread reads through get_history() to stream
// to the workstation history encoder, and it receives the encoder's latent
// embedding through set_embedding().  The design and the row semantics are a
// port of pandapy_dw's HistoryJointPosition adaptor path (Dongwon Son), so
// the existing TAM mapping server / bridge protocol works unchanged.
//
// Only Eigen is required; simadaptor.h (Eigen MLP + weight loader) is a
// verbatim copy of pandapy_dw/include/adaptor/simadaptor.h.

#include <Eigen/Dense>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "simadaptor.h"

namespace rcs {
namespace hw {

class TamHook {
 public:
  using Vec7 = Eigen::Matrix<double, 7, 1>;

  struct HistoryRow {
    double t = 0.0;                    // seconds since the last on_control_start()
    Vec7 q = Vec7::Zero();             // measured joint position
    Vec7 dq = Vec7::Zero();            // measured joint velocity
    Vec7 tau_base = Vec7::Zero();      // RCS law output before the TAM residual (gravity-free)
    Vec7 tau_adaptor_delta = Vec7::Zero();  // TAM residual actually added
    Vec7 tau_applied = Vec7::Zero();   // command returned to libfranka after the RCS tail
    Vec7 tau_commanded = Vec7::Zero(); // robot_state.tau_J_d (last command as processed by libfranka)
    Vec7 tau_measured = Vec7::Zero();  // robot_state.tau_J
    Vec7 gravity = Vec7::Zero();       // franka::Model gravity at this state
    uint64_t history_embedding_seq = 0;  // embedding update used for this row
    bool adaptor_active = false;
    bool valid_for_history = false;    // false for exactly-zero commands / padding
    bool synthetic_padding = false;    // inserted to fill a control-period gap
    bool publish_ready = false;        // all torque fields finalized
    double sample_dt_sec = 0.001;
  };

  struct Status {
    bool loaded = false;
    bool enabled = false;
    bool ideal_model_has_gravity = true;
    uint64_t embedding_seq = 0;
    int embedding_size = 0;
    int expected_embedding_size = 0;
    int history_steps = 0;
    int dof = 7;
    size_t history_size = 0;
    uint64_t rows_total = 0;
    uint64_t start_count = 0;
    double adaptor_forward_dt_ms = 0.0;
    double enable_scale = 0.0;
    std::string last_skip_reason;
    std::string adaptor_path;
    Vec7 torque_limits = Vec7::Zero();
  };

  explicit TamHook(size_t max_history = 4096);

  // ---- non-realtime API (Python side) ---------------------------------
  bool load_adaptor(const std::string& weight_path);
  void set_embedding(const Eigen::VectorXd& embedding);
  uint64_t embedding_seq() const;
  void enable(bool enabled);
  bool enabled() const;
  void set_ideal_model_has_gravity(bool enabled);
  bool ideal_model_has_gravity() const;
  void set_torque_limits(const Vec7& limits);
  void set_enable_ramp_s(double seconds);
  std::vector<HistoryRow> get_history(size_t max_rows) const;
  Status status() const;

  // ---- realtime API (libfranka callback thread) ------------------------
  // Called when a control thread starts: clears history/embedding, resets
  // the time origin and the enable ramp (mirrors pandapy_dw start()).
  void on_control_start();
  // Appends the current row and returns the TAM residual (gravity-free torque
  // units) to be added to tau_base.  Never throws.
  Vec7 apply(double period_sec, const Vec7& q, const Vec7& dq,
             const Vec7& tau_base, const Vec7& gravity,
             const Vec7& tau_commanded, const Vec7& tau_measured);
  // Finalizes the row appended by apply(): stores the applied torque and
  // makes the row visible to get_history().
  void finalize_row(const Vec7& tau_applied);

  // For tests: exact adaptor inputs of the last forward pass (empty if none).
  adaptor::M last_q_hist() const;
  adaptor::M last_dq_hist() const;
  adaptor::M last_tau_hist() const;
  adaptor::M last_embedding_row() const;
  // For tests: inject an in-memory adaptor instead of loading a file.
  void set_adaptor_for_test(std::shared_ptr<adaptor::SimAdaptor> adaptor,
                            const std::string& label = "<in-memory>");

  static constexpr double kHistorySampleDtSec = 0.001;
  static constexpr double kZeroTorqueThresholdNm = 1e-5;
  static constexpr int kMaxPaddingRowsPerStep = 50;

 private:
  double enable_scale_locked(const std::chrono::steady_clock::time_point& now) const;

  mutable std::mutex mux_;
  size_t max_history_;
  std::deque<HistoryRow> history_;
  std::shared_ptr<adaptor::SimAdaptor> adaptor_;
  std::string adaptor_path_;
  bool enabled_ = false;
  std::chrono::steady_clock::time_point enabled_since_;
  std::chrono::steady_clock::time_point start_time_;
  bool started_ = false;
  Eigen::VectorXd embedding_;
  uint64_t embedding_seq_ = 0;
  bool ideal_model_has_gravity_ = true;
  Vec7 torque_limits_;
  double enable_ramp_s_ = 1.0;
  double adaptor_forward_dt_ms_ = 0.0;
  double last_enable_scale_ = 0.0;
  std::string last_skip_reason_ = "not_started";
  uint64_t rows_total_ = 0;
  uint64_t start_count_ = 0;
  Vec7 pending_delta_ = Vec7::Zero();
  bool pending_adaptor_used_ = false;
  adaptor::M last_q_hist_;
  adaptor::M last_dq_hist_;
  adaptor::M last_tau_hist_;
  adaptor::M last_emb_row_;
};

}  // namespace hw
}  // namespace rcs

#endif  // RCS_TAM_HOOK_H
