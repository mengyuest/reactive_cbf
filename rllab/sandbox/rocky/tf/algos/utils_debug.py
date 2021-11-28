import os
import psutil
import numpy as np

def get_memory():
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    return memoryUse # in GB

def style(a, b=None, c=None):
    if c is None:
        if b is None:
            return "%.4f -> %.4f (%.4f)" % (a[0], a[-1], np.mean(a))
        else:
            return "%.4f %.4f -> %.4f %.4f (%.4f %.4f)" % (
                a[0], b[0], a[-1], b[-1], np.mean(a), np.mean(b)
            )
    else:
        return "%.4f %.4f %.4f -> %.4f %.4f %.4f (%.4f %.4f %.4f)" % (
            a[0], b[0], c[0], a[-1], b[-1], c[-1], np.mean(a), np.mean(b), np.mean(c)
        )


def prt(array, sty="%.4f"):
    return "["+" ".join([sty % x for x in array])+"]"

def print_aff(aff, name):
    print(name)
    print("c:%.4f  ll:%.4f rl:%.4f lr:%.4f rr:%.4f"%(aff[0], aff[1], aff[2], aff[3], aff[4]))
    print("th:%.4f v:%.4f  a:%.4f  w:%.4f  L:%.4f  W:%.4f"%(aff[5], aff[6], aff[7], aff[8], aff[9], aff[10]))
    # offset=11
    for i in range(6):
        offset = 11 + 9*i
        print("nei%d-%d x:%.4f y:%.4f th:%.4f v:%.4f a:%.4f w:%.4f L:%.4f W:%.4f" % (
            i, aff[offset], aff[offset+1], aff[offset+2], aff[offset+3], aff[offset+4], aff[offset+5], aff[offset+6], aff[offset+7], aff[offset+8]))
    print("accel [%.7f, %.7f, %.7f, %.7f, %.7f, %.7f]"%tuple([aff[11+9*i+5] for i in range(6)]))
    print("omega [%.7f, %.7f, %.7f, %.7f, %.7f, %.7f]"%tuple([aff[11+9*i+6] for i in range(6)]))
    print()

def debug_in_optimize_policy(np_dict, args):
    # TODO DEBUG
    for ti in range(np_dict["debug_state"].shape[0]):
        if ti > 150:
            break
        sty = "%.5f"
        dbg_ind = [12, 13, 14, 15]
        print("%2d" % ti,
              prt(np_dict["debug_state"][ti, dbg_ind]),
              prt(np_dict["debug_next_state"][ti, dbg_ind]),
              sty % (np_dict["debug_score"][ti]),
              sty % (np_dict["debug_next_score"][ti]),
              prt(np_dict["debug_action"][ti]),
              prt(np_dict["debug_dist_mean"][0, ti]),
              prt(np_dict["debug_ref_action"][0, ti]),
              sty % (np_dict["debug_condition"][ti]),
              int(np_dict["debug_safe"][ti]),
              int(np_dict["debug_medium"][ti]),
              int(np_dict["debug_dang"][ti]))

        print("gradually")
        if args.debug_gradually:
            test_as = [0.5, 0.3, 0.1, 0.05, 0.03, 0.01, -0.01, -0.03, -0.05, -0.1, -0.3, -0.5]
            for dbg_ti, ta in enumerate(test_as):
                print(dbg_ti, "h(a=a%+.4f)=" % ta, np_dict["debug_ha%d" % (dbg_ti)][ti])
        if args.debug_gradually_omega:
            test_as = [0.5, 0.3, 0.1, 0.05, 0.03, 0.01, -0.01, -0.03, -0.05, -0.1, -0.3, -0.5]
            for dbg_ti, ta in enumerate(test_as):
                print(dbg_ti, "h(w=w%+.4f)=" % ta, np_dict["debug_hw%d" % (dbg_ti)][ti])
    # TODO DEBUG(end)
