"""Each live set on today's simulation: which windows each scenario holds, the windows made for
one history has too few of, and the weights that give each its share times its factor (model/scenarios.py)."""

from ...model import scenarios as sm


def for_world(world, sets):
    """Each live set on today's simulation: {"sets" {length: {"set" (live()'s), "a" (World.weights at it), and
    model/scenarios.py at_length's "ext", "w", "inside", "periods", "each"}}, "live" [scenario: {"id"
    (length:scenario), "sid", "horizon", "days", "name", "story", "ranges", "p" (its final probability today),
    "p_data" (the simulation's share today), "p_target" (that times the factor: what it should have; "p" short of it
    only where scenarios are at odds), "times" (the readers' factor), "p_written" and "p_data_written" (both when
    written), "p_history", "readings", "episodes", "made", "starts" (its rows' ids), "node", "set", "written"}]}."""
    out, flat = {}, []
    for h, s in sets.items():
        a = world.weights(s["days"])
        got = sm.at_length(a, s["scenarios"])
        if not got:
            continue
        out[h] = {"set": s, "a": a, **got}
        for x, e in zip(s["scenarios"], got["each"]):
            flat.append(
                {
                    "id": f"{h}:{e['id']}",
                    "sid": e["id"],
                    "horizon": h,
                    "days": s["days"],
                    "name": x["name"],
                    "story": x["story"],
                    "ranges": x["ranges"],
                    "p": e["weight"],
                    "p_data": e["base"],
                    "p_target": e["target"],
                    "times": e["times"],
                    "p_written": x.get("weight"),
                    "p_data_written": x.get("base"),
                    "p_history": e["history"],
                    "readings": x.get("readings", []),
                    "episodes": e["episodes"],
                    "made": e["made"],
                    "starts": e["rows"],
                    "node": s["node"],
                    "set": s["id"],
                    "written": s["written"],
                }
            )
    return {"sets": out, "live": flat}
