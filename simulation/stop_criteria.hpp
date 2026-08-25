#pragma once

/**
 * @file stop_criteria.hpp
 * @brief Criteria for deciding when a run should end.
 *
 * @ref Solver::full_solve takes something it can ask, before every step,
 * whether to stop. This file supplies the useful answers to that question.
 *
 * @section stop_settling Deciding that a scene has settled
 *
 * @ref SettledWhenQuiet watches a quantity that is zero when nothing is moving
 * and stops once that quantity has stayed below a threshold for long enough.
 * Two parameters, and both mean something physical: the threshold says what
 * counts as motion rather than numerical residue, and the required time says
 * how long the quiet has to last before it is believed.
 *
 * A threshold alone would not do. Contact leaves a scene jittering rather than
 * dead, so the measure dips below any reasonable threshold long before the
 * motion is over, and a test that stopped on the first dip would stop during a
 * swing. Requiring the quiet to persist is what distinguishes a lull from an
 * end.
 *
 * @section stop_notabuffer What this does not need
 *
 * No history. The criterion holds one number, the last time the measure was
 * seen above the threshold, and stops when the clock has run far enough past
 * it. That makes the test exact at every step rather than sampled, and costs
 * one comparison per step on top of the measure itself.
 *
 * Measuring every step is affordable: on a scene of three bodies with contact,
 * @ref cosserat::physics::max_material_point_speed costs about a fortieth of
 * one microsecond against a step costing twenty, so under a tenth of one per
 * cent. There is no reason to sample, and sampling would only introduce the
 * risk of missing an excursion above the threshold and stopping early.
 *
 * @section stop_choosing Choosing the numbers
 *
 * The threshold has to sit above the level the scene settles to rather than
 * below it, or the run never stops and always reaches its time limit. That
 * level is a property of the scene: on the meshed example here the material
 * speed comes to rest around a few millimetres per second, so a threshold of a
 * centimetre per second stops it shortly after it visibly stops moving, while
 * a tenth of a millimetre per second never fires.
 *
 * The required time has to exceed the longest period of anything still
 * oscillating. A pendulum's speed passes through zero twice a swing, so a
 * required time shorter than the swing can fall entirely inside a quiet part
 * and call a swinging rod settled.
 *
 * @warning A criterion watching the state cannot promise to return. Every one
 *          here therefore takes a time limit and stops there regardless, and
 *          @ref SettledWhenQuiet::settled tells the two endings apart. They
 *          mean very different things about the state left behind.
 */

#include <cmath>
#include <limits>
#include <utility>

#include "physics/quantities.hpp"

#include "utils/assertions.hpp"

namespace cosserat::simulation {

/**
 * @brief Stops once a measure has stayed below a threshold for long enough.
 *
 * Satisfies the shape @ref Solver::full_solve expects of a stopping criterion.
 * Hold one by value and pass it as an lvalue: the solver takes it by
 * reference, so the state it accumulates survives the run and can be
 * questioned afterwards.
 *
 * @code
 * SettledWhenQuiet criterion(
 *     physics::max_material_point_speed<simulation::SimulationGraph>,
 *     1e-2, 0.25, 10.0);
 * const double reached = solver.full_solve(graph, criterion, 0.0);
 * if (not criterion.settled()) { ... ran out of time ... }
 * @endcode
 *
 * @tparam Measure A callable taking the collection and returning a double that
 *         is zero when nothing is moving.
 */
template<typename Measure>
class SettledWhenQuiet
{
private: // Members
    Measure m_measure;
    double m_threshold;
    double m_required_time_below;
    double m_max_time;

    // The whole of the history this needs: when the measure was last seen
    // above the threshold. Everything else follows from the clock.
    double m_last_above;
    bool m_started;
    bool m_settled;
    double m_last_measured;

public: // Methods
    /**
     * @brief Builds a settling criterion.
     *
     * @param measure What to watch. Called once per step with the collection,
     *        and expected to return a quantity that is zero at rest.
     * @param threshold Below this, the measure counts as residue rather than
     *        motion. Must be finite and not negative. Wants to be above the
     *        level the scene settles to; see @ref stop_choosing.
     * @param required_time_below How long, in simulation time, the measure
     *        must stay below @p threshold before the run is called settled.
     *        Must be finite and positive, and longer than the period of
     *        anything still oscillating.
     * @param max_time Simulation time to stop at whether or not anything
     *        settled. Must be finite.
     */
    SettledWhenQuiet(
        Measure measure,
        double threshold,
        double required_time_below,
        double max_time
    ) : m_measure(std::move(measure)),
        m_threshold(threshold),
        m_required_time_below(required_time_below),
        m_max_time(max_time),
        m_last_above(0.0),
        m_started(false),
        m_settled(false),
        m_last_measured(std::numeric_limits<double>::quiet_NaN())
    {
        utils::nice_assert(
            std::isfinite(threshold) and threshold >= 0.0,
            "the threshold must be finite and not negative"
        );
        utils::nice_assert(
            std::isfinite(required_time_below) and required_time_below > 0.0,
            "required_time_below must be finite and positive"
        );
        utils::nice_assert(
            std::isfinite(max_time), "max_time must be finite"
        );
    }

    /**
     * @brief Asks whether the run should stop.
     *
     * @param system The collection being run.
     * @param time Current simulation time.
     * @return True once the measure has stayed below the threshold for
     *         @c required_time_below, or once @c max_time is reached.
     *
     * @note The clock starts at the first call rather than at construction, so
     *       a criterion built before the run does not count the time it spent
     *       sitting there as quiet.
     *
     * @note The elapsed quiet is compared against the required time as an
     *       ordinary comparison on doubles, so an interval that ought to land
     *       exactly on the requirement can come out a hair under it and the
     *       run takes one more step. Nothing is done about that: a step is far
     *       smaller than any sensible required time, and a tolerance would be
     *       one more arbitrary number to justify.
     */
    template<typename SystemType>
    bool operator()(SystemType& system, double time)
    {
        if (time >= m_max_time) return true;

        // Nothing has been quiet yet at the first call, whatever the measure
        // reads: quiet is a duration, and none has elapsed.
        if (not m_started)
        {
            m_started = true;
            m_last_above = time;
        }

        m_last_measured = m_measure(system);
        if (m_last_measured > m_threshold) m_last_above = time;

        if (time - m_last_above >= m_required_time_below)
        {
            m_settled = true;
            return true;
        }
        return false;
    }

    /**
     * @brief Whether the run ended because the scene settled.
     *
     * @return True if the measure held below the threshold for long enough,
     *         false if the run instead reached its time limit. Worth checking:
     *         @c full_solve reports only a time, and a run that timed out has
     *         left a state that is still moving.
     */
    bool settled() const {return m_settled;}

    /**
     * @brief The most recent value of the measure.
     * @return What the measure last reported, or a quiet NaN if it has not
     *         been asked yet. Useful for reporting how close a run that timed
     *         out came to settling.
     */
    double last_measured() const {return m_last_measured;}

    /**
     * @brief How long the measure has been below the threshold.
     * @param time Current simulation time.
     * @return The quiet elapsed so far, which reaches
     *         @c required_time_below exactly when the run stops.
     */
    double time_below(double time) const
    {
        return m_started ? time - m_last_above : 0.0;
    }

    /** @brief The threshold the measure is compared against. */
    double threshold() const {return m_threshold;}

    /** @brief How long the measure must stay below the threshold. */
    double required_time_below() const {return m_required_time_below;}

    /** @brief The time the run stops at regardless. */
    double max_time() const {return m_max_time;}
};

/**
 * @brief Builds a settling criterion watching how fast anything is moving.
 *
 * The default choice. @ref cosserat::physics::max_material_point_speed is mass
 * independent, so one threshold means the same thing for every body in the
 * scene, and it counts spin, which a nodal speed misses for a rigid body
 * turning in place.
 *
 * @tparam SystemType The collection that will be run.
 * @param threshold Speed below which nothing counts as moving, in metres per
 *        second.
 * @param required_time_below How long it must stay below that, in simulation
 *        time.
 * @param max_time Simulation time to stop at regardless.
 * @return The criterion, to be held by the caller and passed as an lvalue.
 */
template<typename SystemType>
auto settled_when_slow(
    double threshold, double required_time_below, double max_time
)
{
    auto measure = [](SystemType& system)
    {
        return physics::max_material_point_speed(system);
    };
    return SettledWhenQuiet<decltype(measure)>(
        std::move(measure), threshold, required_time_below, max_time);
}

/**
 * @brief Builds a settling criterion watching the total kinetic energy.
 *
 * Prefer @ref settled_when_slow unless there is a reason not to. Energy is
 * quadratic in velocity and linear in mass, so its residual level moves around
 * far more than a speed does as a scene's parameters change, and in a scene of
 * mixed masses a threshold on it is really a statement about the heaviest body
 * present.
 *
 * @tparam SystemType The collection that will be run.
 * @param threshold Energy below which nothing counts as moving, in joules.
 * @param required_time_below How long it must stay below that, in simulation
 *        time.
 * @param max_time Simulation time to stop at regardless.
 * @return The criterion, to be held by the caller and passed as an lvalue.
 */
template<typename SystemType>
auto settled_when_still(
    double threshold, double required_time_below, double max_time
)
{
    auto measure = [](SystemType& system)
    {
        return physics::total_kinetic_energy(system);
    };
    return SettledWhenQuiet<decltype(measure)>(
        std::move(measure), threshold, required_time_below, max_time);
}

/**
 * @brief Stops at a given time and no earlier.
 *
 * What @c full_solve's two time overload uses. Provided by name as well so a
 * caller can hold a criterion either way without the call site changing shape.
 *
 * @param end Simulation time to stop at.
 * @return A criterion that stops there.
 */
inline auto stop_at_time(double end)
{
    utils::nice_assert(std::isfinite(end), "the end time must be finite");
    return [end](auto&, double time) {return time >= end;};
}
} // End namespace cosserat::simulation
