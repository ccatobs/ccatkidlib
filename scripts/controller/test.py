from ccatkidlib.rfsoc.rfsoc_daq import R


def main():
    RC = R()
    RC.find_detectors()
    RC.take_target_sweep()
    RC.take_timestream(60)

if __name__ == "__main__":
    main()