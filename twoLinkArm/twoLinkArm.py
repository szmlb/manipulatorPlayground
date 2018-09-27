import matplotlib.pyplot as plt
import numpy as np
import math

class twoLinkArm:

    def __init__(self, genModelFlag):

        self.step_time = 0.01
        self.dof = 2

        # plot
        plt.ion()
        self.fig = plt.figure()

        self.q = np.zeros(2)
        self.dq = np.zeros(2)
        self.ddq = np.zeros(2)

        # link1
        self.l1 = 0.1
        self.c1 = self.l1 / 2.0
        self.m1 = 0.1
        self.p1x = self.m1 * self.c1
        self.p1y = 0.0
        self.p1z = 0.0
        self.Lxx1 = 0.0
        self.Lxy1 = 0.0
        self.Lxz1 = 0.0
        self.Lyy1 = self.m1 * self.l1**2 / 12.0 + self.m1 * self.c1**2
        self.Lyz1 = 0.0
        self.Lzz1 = self.m1 * self.l1**2 / 12.0 + self.m1 * self.c1**2
        self.jp1 = np.zeros(2)

        # link2
        self.l2 = 0.1
        self.c2 = self.l2 / 2.0
        self.m2 = 0.1
        self.p2x = self.m2 * self.c2
        self.p2y = 0.0
        self.p2z = 0.0
        self.Lxx2 = 0.0
        self.Lxy2 = 0.0
        self.Lxz2 = 0.0
        self.Lyy2 = self.m2 * self.l2**2 / 12.0 + self.m2 * self.c2**2
        self.Lyz2 = 0.0
        self.Lzz2 = self.m2 * self.l2**2 / 12.0 + self.m2 * self.c2**2
        self.jp2 = np.zeros(2)

        # tool center point
        self.tcp = np.zeros(2)

        # equation of motion
        self.parms = [self.Lxx1, self.Lxy1, self.Lxz1, self.Lyy1, self.Lyz1, self.Lzz1, self.p1x, self.p1y, self.p1z, self.m1,
                      self.Lxx2, self.Lxy2, self.Lxz2, self.Lyy2, self.Lyz2, self.Lzz2, self.p2x, self.p2y, self.p2z, self.m2]

        self.Meq = np.zeros([self.dof, self.dof])
        self.ceq = np.zeros([self.dof])
        self.geq = np.zeros([self.dof])
        self.M_matrix(self.parms, self.q)
        self.c_vector(self.parms, self.q, self.dq)
        self.g_vector(self.parms, self.q)

    def M_matrix(self, parms, q):

        M_out = [0]*4

        x0 = 0.1*math.cos(q[1])
        x1 = 0.1*math.sin(q[1])
        x2 = -parms[17]
        x3 = parms[15] + parms[16]*x0 + x1*x2
        M_out[0] = parms[5] + x0*(parms[16] + parms[19]*x0) + x1*(parms[19]*x1 + x2) + x3
        M_out[1] = x3
        M_out[2] = x3
        M_out[3] = parms[15]

        M_out = np.array(M_out)
        self.Meq = M_out.reshape(2, 2)

    def c_vector(self, parms, q, dq):

        c_out = [0]*2

        x0 = -dq[0]**2
        x1 = math.sin(q[1])
        x2 = -0.1*x0*x1
        x3 = math.cos(q[1])
        x4 = 0.1*x0*x3
        x5 = parms[16]*x2 - parms[17]*x4
        x6 = -(dq[0] + dq[1])**2

        c_out[0] = 0.1*x1*(parms[16]*x6 + parms[19]*x4) + 0.1*x3*(parms[17]*x6 + parms[19]*x2) + x5
        c_out[1] = x5

        self.ceq = np.array(c_out)

    def g_vector(self, parms, q):

        g_out = [0]*2

        x0 = 9.81*math.sin(q[0])
        x1 = -x0
        x2 = math.sin(q[1])
        x3 = 9.81*math.cos(q[0])
        x4 = math.cos(q[1])
        x5 = x1*x2 + x3*x4
        x6 = x0*x4 + x2*x3
        x7 = parms[16]*x5 - parms[17]*x6

        g_out[0] = 0.1*parms[19]*x2*x6 + 0.1*parms[19]*x4*x5 + parms[6]*x3 + parms[7]*x1 + x7
        g_out[1] = x7

        self.geq = np.array(g_out)

    def update(self, tau):

        # dynamics
        self.M_matrix(self.parms, self.q)
        self.c_vector(self.parms, self.q, self.dq)
        self.g_vector(self.parms, self.q)

        self.ddq = np.linalg.inv(self.Meq).dot( tau - self.ceq - self.geq  )
        self.dq = self.dq + self.ddq * self.step_time
        self.q = self.q + self.dq * self.step_time

        # kinematics
        self.jp1[0] = 0.0
        self.jp1[1] = 0.0

        self.jp2[0] = self.jp1[0] + self.l1 * np.cos(self.q[0])
        self.jp2[1] = self.jp1[1] + self.l1 * np.sin(self.q[0])

        self.tcp[0] = self.jp1[0] + self.jp2[0] + self.l2 * np.cos(self.q[0] + self.q[1])
        self.tcp[1] = self.jp1[0] + self.jp2[1] + self.l2 * np.sin(self.q[0] + self.q[1])

    def simulation(self, sim_time):

        for i in range(int(sim_time/self.step_time)):

            time = i * self.step_time

            tau = [0.0,  0.0]
            self.update(tau)
            self.plot()

    def plot(self):

        # プロット
        plt.cla() # 現在のプロットを削除

        plt.plot([self.jp1[0], self.jp2[0]], [self.jp1[1], self.jp2[1]], 'k-') # リンク1の直線
        plt.plot([self.jp2[0], self.tcp[0]], [self.jp2[1], self.tcp[1]], 'k-') # リンク2の直線

        plt.plot(self.jp1[0], self.jp1[1], 'ro') # 関節1の位置
        plt.plot(self.jp2[0], self.jp2[1], 'ro') # 関節2の位置
        plt.plot(self.tcp[0], self.tcp[1], 'ro') # 手先の位置

        plt.xlim(-0.2, 0.2) # xlimit
        plt.ylim(-0.3, 0.1) # ylimit

        plt.show()
        plt.pause(self.step_time)

def main():

    rob = twoLinkArm(genModelFlag=False)
    rob.simulation(5.0)

if __name__ == "__main__":
    main()
