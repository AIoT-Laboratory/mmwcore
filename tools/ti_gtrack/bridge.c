/* Host allocation and observation only; all numerical tracking runs in unmodified TI sources. */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ti/alg/gtrack/gtrack.h>
#include <ti/alg/gtrack/include/gtrack_int.h>
#include "bridge.h"

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

/* Pinned SDK allocates ceil(N/8) bitmap bytes but clears (N>>3)+1 in step.
 * Reserve one zeroed guard byte in the host allocator, including when N%8==0.
 * This changes memory allocation only; the TI numerical sources stay intact. */
void *gtrack_alloc(uint32_t count, uint32_t size) {
    if (count && (size_t)size > (SIZE_MAX - 1) / count) return NULL;
    return calloc(1, (size_t)count * size + 1);
}
void gtrack_free(void *ptr, uint32_t size) { (void)size; free(ptr); }
void gtrack_log(GTRACK_VERBOSE_TYPE level, const char *format, ...) { (void)level; (void)format; }

typedef struct {
    void *module;
    uint32_t max_points, max_tracks;
    GTRACK_measurementPoint *points;
    GTRACK_measurement_vector *variances;
    GTRACK_targetDesc *targets;
    uint8_t *indices, *unique;
} Host;

EXPORT uint32_t mmw_ti_abi(uint32_t kind) {
    if (kind == 0) return 1;
    if (kind == 1) return (uint32_t)sizeof(MmwTiConfig);
    if (kind == 2) return (uint32_t)sizeof(MmwTiTarget);
    return 0;
}

EXPORT void mmw_ti_delete(void *handle) {
    Host *host = handle;
    if (!host) return;
    if (host->module) gtrack_delete(host->module);
    free(host->points); free(host->variances); free(host->targets);
    free(host->indices); free(host->unique); free(host);
}

EXPORT void *mmw_ti_create(const MmwTiConfig *c, int32_t *error) {
    Host *host = calloc(1, sizeof(*host));
    GTRACK_moduleConfig config = {0};
    GTRACK_advancedParameters advanced = {0};
    GTRACK_gatingParams gating = {0};
    GTRACK_allocationParams allocation = {0};
    GTRACK_stateParams state = {0};
    GTRACK_sceneryParams scenery = {0};
    GTRACK_presenceParams presence = {0};
    *error = -1;
    if (!host || !c || c->max_points < 1 || c->max_points > GTRACK_NUM_POINTS_MAX ||
        c->max_tracks < 1 || c->max_tracks > GTRACK_NUM_TRACKS_MAX ||
        c->boundary_count > 2 || c->static_count > 2 || c->occupancy_count > 2) {
        free(host); return NULL;
    }
    host->max_points = c->max_points; host->max_tracks = c->max_tracks;
    host->points = calloc(c->max_points, sizeof(*host->points));
    host->variances = calloc(c->max_points, sizeof(*host->variances));
    host->targets = calloc(c->max_tracks, sizeof(*host->targets));
    host->indices = calloc(c->max_points, 1);
    host->unique = calloc((c->max_points >> 3) + 1, 1);
    if (!host->points || !host->variances || !host->targets || !host->indices || !host->unique) {
        mmw_ti_delete(host); return NULL;
    }
    config.stateVectorType = GTRACK_STATE_VECTORS_3DA;
    config.verbose = GTRACK_VERBOSE_NONE;
    config.maxNumPoints = (uint16_t)c->max_points; config.maxNumTracks = (uint16_t)c->max_tracks;
    config.deltaT = c->delta_t; config.initialRadialVelocity = c->initial_velocity;
    config.maxRadialVelocity = c->max_velocity; config.radialVelocityResolution = c->velocity_resolution;
    memcpy(config.maxAcceleration, c->max_acceleration, sizeof(config.maxAcceleration));
    config.boresightFilteringEnable = (uint16_t)c->boresight_filtering;
    gating.gain = c->gating_gain;
    memcpy(gating.limitsArray, c->gating_limits, sizeof(gating.limitsArray));
    allocation.snrThre = c->allocation_snr; allocation.snrThreObscured = c->allocation_obscured_snr;
    allocation.velocityThre = c->allocation_velocity; allocation.pointsThre = (uint16_t)c->allocation_points;
    allocation.maxDistanceThre = c->allocation_distance; allocation.maxVelThre = c->allocation_max_velocity;
    state.det2actThre = (uint16_t)c->state_thresholds[0]; state.det2freeThre = (uint16_t)c->state_thresholds[1];
    state.active2freeThre = (uint16_t)c->state_thresholds[2]; state.static2freeThre = (uint16_t)c->state_thresholds[3];
    state.exit2freeThre = (uint16_t)c->state_thresholds[4]; state.sleep2freeThre = (uint16_t)c->state_thresholds[5];
    memcpy(&scenery.sensorPosition, c->sensor_position, sizeof(scenery.sensorPosition));
    memcpy(&scenery.sensorOrientation, c->sensor_orientation, sizeof(scenery.sensorOrientation));
    scenery.numBoundaryBoxes = (uint8_t)c->boundary_count; scenery.numStaticBoxes = (uint8_t)c->static_count;
    memcpy(scenery.boundaryBox, c->boundary_boxes, sizeof(scenery.boundaryBox));
    memcpy(scenery.staticBox, c->static_boxes, sizeof(scenery.staticBox));
    presence.pointsThre = (uint16_t)c->presence_points; presence.velocityThre = c->presence_velocity;
    presence.on2offThre = (uint16_t)c->presence_on_to_off; presence.numOccupancyBoxes = (uint8_t)c->occupancy_count;
    memcpy(presence.occupancyBox, c->occupancy_boxes, sizeof(presence.occupancyBox));
    advanced.gatingParams = &gating; advanced.allocationParams = &allocation; advanced.stateParams = &state;
    advanced.sceneryParams = &scenery; advanced.presenceParams = &presence; config.advParams = &advanced;
    host->module = gtrack_create(&config, error);
    if (!host->module) { mmw_ti_delete(host); return NULL; }
    return host;
}

/* The input is copied because gtrack_step may unroll its point Doppler in place. */
EXPORT int32_t mmw_ti_step(void *handle, const float *points, const float *variances, uint32_t count,
                          MmwTiTarget *targets, uint32_t *target_count, uint8_t *indices,
                          uint8_t *unique, uint8_t *static_flags, float *scores,
                          float *updated_doppler, uint32_t *presence, uint32_t *bench) {
    Host *host = handle;
    if (!host || count > host->max_points) return -1;
    for (uint32_t i = 0; i < count; i++) {
        memcpy(host->points[i].array, points + i * 5, 4 * sizeof(float));
        host->points[i].snr = points[i * 5 + 4];
        if (variances) memcpy(&host->variances[i], variances + i * 4, 4 * sizeof(float));
    }
    uint16_t number = 0;
    uint8_t detected = 0;
    gtrack_step(host->module, host->points, variances ? host->variances : NULL, (uint16_t)count,
                host->targets, &number, host->indices, host->unique, &detected, bench);
    GtrackModuleInstance *module = host->module;
    *target_count = number; *presence = detected;
    for (uint32_t i = 0; i < number; i++) {
        const GTRACK_targetDesc *src = &host->targets[i];
        const GtrackUnitInstance *unit = module->hTrack[src->uid];
        MmwTiTarget *dst = &targets[i];
        memset(dst, 0, sizeof(*dst));
        dst->uid = src->uid; dst->tid = src->tid; dst->state = unit->state;
        dst->velocity_state = unit->velocityHandling; dst->is_static = unit->isTargetStatic;
        dst->snr_weighting = unit->isSnrWeighting; dst->height_ignore = unit->isAssociationHeightIgnore;
        dst->point_number_estimation = unit->isEnablePointNumberEstimation;
        dst->counters[0] = unit->detect2activeCount; dst->counters[1] = unit->detect2freeCount;
        dst->counters[2] = unit->active2freeCount; dst->counters[3] = unit->sleep2freeCount;
        dst->counters[4] = unit->outside2freeCount; dst->counters[5] = unit->numStaticPtsHistory;
        dst->age = unit->heartBeatCount - unit->allocationTime + 1;
        memcpy(dst->state_vector, src->S, sizeof(dst->state_vector));
        memcpy(dst->state_covariance, unit->P_hat, sizeof(dst->state_covariance));
        memcpy(dst->predicted_state, unit->S_apriori_hat, sizeof(dst->predicted_state));
        memcpy(dst->predicted_covariance, unit->P_apriori_hat, sizeof(dst->predicted_covariance));
        memcpy(dst->predicted_measurement, unit->H_s.array, sizeof(dst->predicted_measurement));
        memcpy(dst->ec, src->EC, sizeof(dst->ec));
        memcpy(dst->group_covariance, unit->gC, sizeof(dst->group_covariance));
        memcpy(dst->group_dispersion, unit->gD, sizeof(dst->group_dispersion));
        memcpy(dst->dimensions, src->dim, sizeof(dst->dimensions));
        memcpy(dst->measurement_center, src->uCenter, sizeof(dst->measurement_center));
        dst->gain = src->G; dst->confidence = src->confidenceLevel;
        dst->expected_points = unit->estNumOfPoints; dst->range_rate = unit->rangeRate;
    }
    for (uint32_t i = 0; i < count; i++) {
        indices[i] = host->indices[i]; unique[i] = (host->unique[i >> 3] >> (i & 7)) & 1;
        static_flags[i] = (module->isStaticIndex[i >> 3] >> (i & 7)) & 1;
        scores[i] = module->bestScore[i]; updated_doppler[i] = host->points[i].vector.doppler;
    }
    return 0;
}
