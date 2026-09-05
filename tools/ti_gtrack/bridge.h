/* mmwcore host ABI. The separately installed TI implementation is not bundled. */
#ifndef MMWCORE_TI_GTRACK_BRIDGE_H
#define MMWCORE_TI_GTRACK_BRIDGE_H
#include <stdint.h>

typedef struct {
    uint32_t max_points, max_tracks;
    float delta_t, initial_velocity, max_velocity, velocity_resolution;
    float max_acceleration[3];
    uint32_t boresight_filtering;
    float gating_gain, gating_limits[4];
    float allocation_snr, allocation_obscured_snr, allocation_velocity;
    uint32_t allocation_points;
    float allocation_distance, allocation_max_velocity;
    uint32_t state_thresholds[6];
    float sensor_position[3], sensor_orientation[2];
    uint32_t boundary_count, static_count, occupancy_count;
    float boundary_boxes[12], static_boxes[12], occupancy_boxes[12];
    uint32_t presence_points, presence_on_to_off;
    float presence_velocity;
} MmwTiConfig;

typedef struct {
    uint32_t uid, tid, state, velocity_state;
    uint32_t is_static, snr_weighting, height_ignore, point_number_estimation;
    uint32_t counters[6]; /* detect2active, detect2free, active2free, sleep2free, outside2free, static history */
    uint64_t age;
    float state_vector[9], state_covariance[81];
    float predicted_state[9], predicted_covariance[81], predicted_measurement[4];
    float ec[16], group_covariance[16], group_dispersion[16];
    float gain, dimensions[4], measurement_center[4], confidence;
    float expected_points, range_rate;
} MmwTiTarget;

#endif
