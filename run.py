import matplotlib.pyplot as plt
import numpy as np

# Similation parameters
dt = 0.01

# Link lengths
l1 = l2 = 1

# Set initial goal position to the initial end-effector position
x = 2
y = 0

def simulation(GOAL_TH=0.0, theta1=0.0, theta2=0.0):

    for i in range(int(2.0/dt)):

        time = i * dt

        theta1 = 0.1 * np.sin(2.0 * np.pi * time)
        theta2 = 0.1 * np.sin(2.0 * np.pi * time)

        x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
        y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)

        wrist = plot_arm(theta1, theta2, x, y)

def plot_arm(theta1, theta2, x, y):

    # 各関節の位置
    shoulder = np.array([0, 0])
    elbow = shoulder + np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)])
    wrist = elbow + np.array([l2 * np.cos(theta1 + theta2), l2 * np.sin(theta1 + theta2)])

    # プロット
    plt.cla()

    plt.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], 'k-') # リンク1の直線
    plt.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], 'k-')       # リンク2の直線

    plt.plot(shoulder[0], shoulder[1], 'ro') # 関節1の位置
    plt.plot(elbow[0], elbow[1], 'ro')       # 関節2の位置
    plt.plot(wrist[0], wrist[1], 'ro')       # 手先の位置

    plt.plot([wrist[0], x], [wrist[1], y], 'g--') # 目標位置との直線
    plt.plot(x, y, 'g*') # 手先の星印

    plt.xlim(-2, 2) # xlimit
    plt.ylim(-2, 2) # ylimit

    plt.show()
    plt.pause(dt)

    return wrist

def main():

    plt.ion()
    fig = plt.figure()
    simulation()

if __name__ == "__main__":
    main()
