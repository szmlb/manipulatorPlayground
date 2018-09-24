import matplotlib.pyplot as plt
import twoLinkArm.twoLinkArm as tla

def main():

    plt.ion()
    fig = plt.figure()
    rob = tla.twoLinkArm(genModelFlag=False)
    rob.simulation(2.0)

if __name__ == "__main__":
    main()
