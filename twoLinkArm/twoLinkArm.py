import matplotlib.pyplot as plt
import numpy as np

class twoLinkArm:

    def __init__(self, genModelFlag):
        self.step_time = 0.01
        self.l1 = 0.1
        self.l2 = 0.1
        self.tcp_x = 0.2
        self.tcp_z = 0.0

    def simulation(self, sim_time):

        for i in range(int(sim_time/self.step_time)):

            time = i * self.step_time

            self.theta1 = self.l1 * np.sin(2.0 * np.pi * time)
            self.theta2 = self.l2 * np.sin(2.0 * np.pi * time)

            self.tcp_x = self.l1 * np.cos(self.theta1) + self.l2 * np.cos(self.theta1 + self.theta2)
            self.tcp_z = self.l1 * np.sin(self.theta1) + self.l2 * np.sin(self.theta1 + self.theta2)

            wrist = self.plot_arm()

    def plot_arm(self):

        # 各関節の位置
        shoulder = np.array([0, 0])
        elbow = shoulder + np.array([self.l1 * np.cos(self.theta1), self.l1 * np.sin(self.theta1)])
        wrist = elbow + np.array([self.l2 * np.cos(self.theta1 + self.theta2), self.l2 * np.sin(self.theta1 + self.theta2)])

        # プロット
        plt.cla()

        plt.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], 'k-') # リンク1の直線
        plt.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], 'k-')       # リンク2の直線

        plt.plot(shoulder[0], shoulder[1], 'ro') # 関節1の位置
        plt.plot(elbow[0], elbow[1], 'ro')       # 関節2の位置
        plt.plot(wrist[0], wrist[1], 'ro')       # 手先の位置

        plt.plot([wrist[0], self.tcp_x], [wrist[1], self.tcp_z], 'g--') # 目標位置との直線
        plt.plot(self.tcp_x, self.tcp_z, 'g*') # 手先の星印

        plt.xlim(-0.2, 0.2) # xlimit
        plt.ylim(-0.2, 0.2) # ylimit

        plt.show()
        plt.pause(self.step_time)

        return wrist

def main():

    plt.ion()
    fig = plt.figure()
    rob = twoLinkArm(genModelFlag=False)
    rob.simulation(2.0)

if __name__ == "__main__":
    main()
