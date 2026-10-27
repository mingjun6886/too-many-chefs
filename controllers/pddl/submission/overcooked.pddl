; Instance-level kitchen model.
;
; Every chef, every tile a chef acts on (a "slot": counter, station,
; storage, delivery spot or bin), every piece of food, every portable piece
; of equipment (a "vessel": plate, pan, pot, mixing bowl) and every visible
; order is an object. Food keeps its identity while it is cooked or
; combined: its current kind is the fluent (kind ?f ?k). When two foods
; combine, one object carries the result and the other disappears.
; Storages hand out "token" food objects that do not exist yet: (spawn ?f ?k)
; says token ?f becomes a fresh ?k when taken.
;
; Recipes and food rules are static facts generated from the kitchen:
; (cooks ?e ?in ?out), (combines ?a ?b ?out) (listed both ways round),
; (raw ?k) for food that may sit on a bare counter, and (accepts ?e ?k) when
; empty equipment of kind ?e takes food of kind ?k.
;
; Movement is not modelled. (reach ?c ?s) says chef ?c can walk next to
; slot ?s; the controller walks the chef there before each step.
;
; Every action names the acting chef first and the slot it acts on second;
; from_pddl.py relies on that order.
(define (domain overcooked)
  (:requirements :strips :typing :negative-preconditions)

  (:types
    chef slot item fkind ekind order - object
    food vessel - item
  )

  (:predicates
    ; Chefs.
    (free ?c - chef)
    (holding ?c - chef ?i - item)
    (reach ?c - chef ?s - slot)
    ; Slots. A slot holds at most one item; (clear ?s) when it holds none.
    (counter ?s - slot)
    (station ?s - slot ?e - ekind)
    (storage ?s - slot ?k - fkind)
    (delivery ?s - slot)
    (bin ?s - slot)
    (clear ?s - slot)
    (at ?i - item ?s - slot)
    ; Vessels. A vessel holds at most one food; (empty ?v) when it holds none.
    (vkind ?v - vessel ?e - ekind)
    (servable ?v - vessel)
    (empty ?v - vessel)
    (in ?f - food ?v - vessel)
    ; Food.
    (kind ?f - food ?k - fkind)
    (spawn ?f - food ?k - fkind)
    ; Recipes and rules.
    (raw ?k - fkind)
    (cooks ?e - ekind ?in - fkind ?out - fkind)
    (manual ?e - ekind)
    (combines ?a - fkind ?b - fkind ?out - fkind)
    (accepts ?e - ekind ?k - fkind)
    (requires ?e - ekind ?base - ekind)
    ; Orders.
    (dish ?o - order ?k - fkind)
    (served ?o - order)
  )

  ; --- Storage -----------------------------------------------------------

  ; Take a fresh ingredient into empty hands.
  (:action take
    :parameters (?c - chef ?s - slot ?f - food ?k - fkind)
    :precondition (and (free ?c) (reach ?c ?s) (storage ?s ?k) (spawn ?f ?k))
    :effect (and (not (free ?c)) (holding ?c ?f) (kind ?f ?k) (not (spawn ?f ?k))))

  ; Take a fresh ingredient straight into a held, empty vessel.
  (:action take_into
    :parameters (?c - chef ?s - slot ?v - vessel ?e - ekind ?f - food ?k - fkind)
    :precondition (and (holding ?c ?v) (reach ?c ?s) (storage ?s ?k) (spawn ?f ?k)
                       (vkind ?v ?e) (empty ?v) (accepts ?e ?k))
    :effect (and (in ?f ?v) (kind ?f ?k) (not (empty ?v)) (not (spawn ?f ?k))))

  ; Take an ingredient that combines with the food in a held vessel.
  (:action take_mix
    :parameters (?c - chef ?s - slot ?v - vessel ?e - ekind ?g - food ?kg - fkind
                 ?k - fkind ?out - fkind)
    :precondition (and (holding ?c ?v) (reach ?c ?s) (storage ?s ?k)
                       (vkind ?v ?e) (in ?g ?v) (kind ?g ?kg)
                       (combines ?k ?kg ?out) (accepts ?e ?out))
    :effect (and (not (kind ?g ?kg)) (kind ?g ?out)))

  ; --- Picking up and putting down -------------------------------------------

  ; Pick up food from a counter or out of a station.
  (:action pick_up
    :parameters (?c - chef ?s - slot ?f - food ?k - fkind)
    :precondition (and (free ?c) (reach ?c ?s) (at ?f ?s) (kind ?f ?k))
    :effect (and (not (free ?c)) (holding ?c ?f) (not (at ?f ?s)) (clear ?s)))

  ; Pick up an empty vessel.
  (:action pick_up_vessel
    :parameters (?c - chef ?s - slot ?v - vessel)
    :precondition (and (free ?c) (reach ?c ?s) (at ?v ?s) (empty ?v))
    :effect (and (not (free ?c)) (holding ?c ?v) (not (at ?v ?s)) (clear ?s)))

  ; Pick up a vessel with the food in it.
  (:action pick_up_full
    :parameters (?c - chef ?s - slot ?v - vessel ?f - food ?k - fkind)
    :precondition (and (free ?c) (reach ?c ?s) (at ?v ?s) (in ?f ?v) (kind ?f ?k))
    :effect (and (not (free ?c)) (holding ?c ?v) (not (at ?v ?s)) (clear ?s)))

  ; Put raw food on a bare counter.
  (:action put_food
    :parameters (?c - chef ?s - slot ?f - food ?k - fkind)
    :precondition (and (holding ?c ?f) (reach ?c ?s) (counter ?s) (clear ?s)
                       (kind ?f ?k) (raw ?k))
    :effect (and (free ?c) (not (holding ?c ?f)) (at ?f ?s) (not (clear ?s))))

  ; Put a vessel, with whatever is in it, on a bare counter.
  (:action put_vessel
    :parameters (?c - chef ?s - slot ?v - vessel)
    :precondition (and (holding ?c ?v) (reach ?c ?s) (counter ?s) (clear ?s))
    :effect (and (free ?c) (not (holding ?c ?v)) (at ?v ?s) (not (clear ?s))))

  ; Set a vessel on the station it works on, such as a pan on a stove.
  (:action mount
    :parameters (?c - chef ?s - slot ?v - vessel ?e - ekind ?b - ekind)
    :precondition (and (holding ?c ?v) (reach ?c ?s) (station ?s ?b) (clear ?s)
                       (vkind ?v ?e) (requires ?e ?b))
    :effect (and (free ?c) (not (holding ?c ?v)) (at ?v ?s) (not (clear ?s))))

  ; --- Held food into or onto something ------------------------------------

  ; Put held food into an empty station.
  (:action load
    :parameters (?c - chef ?s - slot ?e - ekind ?f - food ?k - fkind)
    :precondition (and (holding ?c ?f) (reach ?c ?s) (station ?s ?e) (clear ?s)
                       (kind ?f ?k) (accepts ?e ?k))
    :effect (and (free ?c) (not (holding ?c ?f)) (at ?f ?s) (not (clear ?s))))

  ; Add held food to the food in a station; the station keeps the result.
  (:action load_mix
    :parameters (?c - chef ?s - slot ?e - ekind ?f - food ?kf - fkind
                 ?g - food ?kg - fkind ?out - fkind)
    :precondition (and (holding ?c ?f) (reach ?c ?s) (station ?s ?e) (at ?g ?s)
                       (kind ?f ?kf) (kind ?g ?kg) (combines ?kf ?kg ?out)
                       (accepts ?e ?out))
    :effect (and (free ?c) (not (holding ?c ?f)) (not (kind ?g ?kg)) (kind ?g ?out)))

  ; Combine held food with the food in a station that will not take the
  ; result; the chef ends up holding it.
  (:action grab_mix
    :parameters (?c - chef ?s - slot ?e - ekind ?f - food ?kf - fkind
                 ?g - food ?kg - fkind ?out - fkind)
    :precondition (and (holding ?c ?f) (reach ?c ?s) (station ?s ?e) (at ?g ?s)
                       (kind ?f ?kf) (kind ?g ?kg) (combines ?kf ?kg ?out)
                       (not (accepts ?e ?out)))
    :effect (and (not (at ?g ?s)) (clear ?s) (not (kind ?f ?kf)) (kind ?f ?out)))

  ; Put held food into an empty vessel standing on a slot.
  (:action fill
    :parameters (?c - chef ?s - slot ?v - vessel ?e - ekind ?f - food ?k - fkind)
    :precondition (and (holding ?c ?f) (reach ?c ?s) (at ?v ?s) (vkind ?v ?e)
                       (empty ?v) (kind ?f ?k) (accepts ?e ?k))
    :effect (and (free ?c) (not (holding ?c ?f)) (in ?f ?v) (not (empty ?v))))

  ; Add held food to the food in a vessel standing on a slot.
  (:action fill_mix
    :parameters (?c - chef ?s - slot ?v - vessel ?e - ekind ?f - food ?kf - fkind
                 ?g - food ?kg - fkind ?out - fkind)
    :precondition (and (holding ?c ?f) (reach ?c ?s) (at ?v ?s) (vkind ?v ?e)
                       (in ?g ?v) (kind ?f ?kf) (kind ?g ?kg)
                       (combines ?kf ?kg ?out) (accepts ?e ?out))
    :effect (and (free ?c) (not (holding ?c ?f)) (not (kind ?g ?kg)) (kind ?g ?out)))

  ; Combine held food with the food in a vessel that will not take the
  ; result, such as a bun with the patty in a pan; the chef holds the result.
  (:action grab_from_vessel
    :parameters (?c - chef ?s - slot ?v - vessel ?e - ekind ?f - food ?kf - fkind
                 ?g - food ?kg - fkind ?out - fkind)
    :precondition (and (holding ?c ?f) (reach ?c ?s) (at ?v ?s) (vkind ?v ?e)
                       (in ?g ?v) (kind ?f ?kf) (kind ?g ?kg)
                       (combines ?kf ?kg ?out) (not (accepts ?e ?out)))
    :effect (and (not (in ?g ?v)) (empty ?v) (not (kind ?f ?kf)) (kind ?f ?out)))

  ; Combine held food with raw food on a bare counter. The result stays on
  ; the counter, so it has to be raw too.
  (:action merge
    :parameters (?c - chef ?s - slot ?f - food ?kf - fkind
                 ?g - food ?kg - fkind ?out - fkind)
    :precondition (and (holding ?c ?f) (reach ?c ?s) (counter ?s) (at ?g ?s)
                       (kind ?f ?kf) (kind ?g ?kg) (combines ?kf ?kg ?out) (raw ?out))
    :effect (and (free ?c) (not (holding ?c ?f)) (not (kind ?g ?kg)) (kind ?g ?out)))

  ; --- Held vessel collects or pours ----------------------------------------

  ; Collect food from a counter or station into a held, empty vessel.
  (:action collect
    :parameters (?c - chef ?s - slot ?v - vessel ?e - ekind ?f - food ?k - fkind)
    :precondition (and (holding ?c ?v) (reach ?c ?s) (vkind ?v ?e) (empty ?v)
                       (at ?f ?s) (kind ?f ?k) (accepts ?e ?k))
    :effect (and (not (at ?f ?s)) (clear ?s) (in ?f ?v) (not (empty ?v))))

  ; Collect food from a counter into a held vessel, combining it with the
  ; food already there.
  (:action collect_mix
    :parameters (?c - chef ?s - slot ?v - vessel ?e - ekind ?g - food ?kg - fkind
                 ?f - food ?kf - fkind ?out - fkind)
    :precondition (and (holding ?c ?v) (reach ?c ?s) (counter ?s) (vkind ?v ?e)
                       (in ?g ?v) (kind ?g ?kg) (at ?f ?s) (kind ?f ?kf)
                       (combines ?kf ?kg ?out) (accepts ?e ?out))
    :effect (and (not (at ?f ?s)) (clear ?s) (not (kind ?g ?kg)) (kind ?g ?out)))

  ; The same from a station. The simulator first tries to pour into the
  ; station, so this only happens when the station will not take the result.
  (:action collect_mix_station
    :parameters (?c - chef ?s - slot ?se - ekind ?v - vessel ?e - ekind
                 ?g - food ?kg - fkind ?f - food ?kf - fkind ?out - fkind)
    :precondition (and (holding ?c ?v) (reach ?c ?s) (station ?s ?se) (vkind ?v ?e)
                       (in ?g ?v) (kind ?g ?kg) (at ?f ?s) (kind ?f ?kf)
                       (combines ?kf ?kg ?out) (accepts ?e ?out)
                       (not (accepts ?se ?out)))
    :effect (and (not (at ?f ?s)) (clear ?s) (not (kind ?g ?kg)) (kind ?g ?out)))

  ; Pour the food in a held vessel into an empty vessel on a slot, such as
  ; a grilled patty from the pan onto a plate. The chef keeps the vessel.
  (:action pour
    :parameters (?c - chef ?s - slot ?v - vessel ?f - food ?k - fkind
                 ?w - vessel ?ew - ekind)
    :precondition (and (holding ?c ?v) (reach ?c ?s) (in ?f ?v) (kind ?f ?k)
                       (at ?w ?s) (vkind ?w ?ew) (empty ?w) (accepts ?ew ?k))
    :effect (and (not (in ?f ?v)) (empty ?v) (in ?f ?w) (not (empty ?w))))

  ; Pour into a vessel that holds food the poured food combines with.
  (:action pour_mix
    :parameters (?c - chef ?s - slot ?v - vessel ?f - food ?kf - fkind
                 ?w - vessel ?ew - ekind ?g - food ?kg - fkind ?out - fkind)
    :precondition (and (holding ?c ?v) (reach ?c ?s) (in ?f ?v) (kind ?f ?kf)
                       (at ?w ?s) (vkind ?w ?ew) (in ?g ?w) (kind ?g ?kg)
                       (combines ?kf ?kg ?out) (accepts ?ew ?out))
    :effect (and (not (in ?f ?v)) (empty ?v) (not (kind ?g ?kg)) (kind ?g ?out)))

  ; Pour the food in a held vessel into an empty station, such as batter
  ; from the mixing bowl into the oven.
  (:action pour_station
    :parameters (?c - chef ?s - slot ?e - ekind ?v - vessel ?f - food ?k - fkind)
    :precondition (and (holding ?c ?v) (reach ?c ?s) (station ?s ?e) (clear ?s)
                       (in ?f ?v) (kind ?f ?k) (accepts ?e ?k))
    :effect (and (not (in ?f ?v)) (empty ?v) (at ?f ?s) (not (clear ?s))))

  ; Scoop the food out of a vessel on a slot into a held, empty vessel,
  ; such as a grilled patty from the pan onto a held plate.
  (:action scoop
    :parameters (?c - chef ?s - slot ?v - vessel ?ev - ekind ?w - vessel
                 ?f - food ?k - fkind)
    :precondition (and (holding ?c ?v) (reach ?c ?s) (vkind ?v ?ev) (empty ?v)
                       (at ?w ?s) (in ?f ?w) (kind ?f ?k) (accepts ?ev ?k))
    :effect (and (not (in ?f ?w)) (empty ?w) (in ?f ?v) (not (empty ?v))))

  ; Scoop into a held vessel whose food combines with it. The simulator first
  ; tries to pour the other way, so the vessel on the slot must refuse the
  ; result.
  (:action scoop_mix
    :parameters (?c - chef ?s - slot ?v - vessel ?ev - ekind ?g - food ?kg - fkind
                 ?w - vessel ?ew - ekind ?f - food ?kf - fkind ?out - fkind)
    :precondition (and (holding ?c ?v) (reach ?c ?s) (vkind ?v ?ev) (in ?g ?v)
                       (kind ?g ?kg) (at ?w ?s) (vkind ?w ?ew) (in ?f ?w)
                       (kind ?f ?kf) (combines ?kf ?kg ?out) (accepts ?ev ?out)
                       (not (accepts ?ew ?out)))
    :effect (and (not (in ?f ?w)) (empty ?w) (not (kind ?g ?kg)) (kind ?g ?out)))

  ; --- Cooking --------------------------------------------------------------

  ; Work the food in a station, such as chopping on a cutboard.
  (:action cook
    :parameters (?c - chef ?s - slot ?e - ekind ?f - food ?in - fkind ?out - fkind)
    :precondition (and (reach ?c ?s) (station ?s ?e) (manual ?e) (at ?f ?s)
                       (kind ?f ?in) (cooks ?e ?in ?out))
    :effect (and (not (kind ?f ?in)) (kind ?f ?out)))

  ; Work the food in a vessel standing on its base, such as a pan on a stove.
  (:action cook_vessel
    :parameters (?c - chef ?s - slot ?v - vessel ?e - ekind ?b - ekind ?f - food
                 ?in - fkind ?out - fkind)
    :precondition (and (reach ?c ?s) (at ?v ?s) (vkind ?v ?e) (station ?s ?b)
                       (requires ?e ?b) (manual ?e)
                       (in ?f ?v) (kind ?f ?in) (cooks ?e ?in ?out))
    :effect (and (not (kind ?f ?in)) (kind ?f ?out)))

  ; --- Serving and clearing up ----------------------------------------------

  ; Serve a plated dish for an order that has not been served yet.
  (:action deliver
    :parameters (?c - chef ?s - slot ?v - vessel ?f - food ?k - fkind ?o - order)
    :precondition (and (holding ?c ?v) (reach ?c ?s) (delivery ?s) (servable ?v)
                       (in ?f ?v) (kind ?f ?k) (dish ?o ?k) (not (served ?o)))
    :effect (and (served ?o) (free ?c) (not (holding ?c ?v)) (not (in ?f ?v))))

  ; Throw held food in the bin.
  (:action discard
    :parameters (?c - chef ?s - slot ?f - food ?k - fkind)
    :precondition (and (holding ?c ?f) (reach ?c ?s) (bin ?s) (kind ?f ?k))
    :effect (and (free ?c) (not (holding ?c ?f))))

  ; Empty a held vessel into the bin; the chef keeps the vessel.
  (:action discard_from
    :parameters (?c - chef ?s - slot ?v - vessel ?f - food ?k - fkind)
    :precondition (and (holding ?c ?v) (reach ?c ?s) (bin ?s) (in ?f ?v) (kind ?f ?k))
    :effect (and (not (in ?f ?v)) (empty ?v)))
)
