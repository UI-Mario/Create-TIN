import numpy as np
from delaunay2D import Delaunay2D

if __name__ == '__main__':
    radius = 100

    P = np.loadtxt('facePoints.xyztri')
    # 二维点
    seeds = np.zeros((len(P), 2))
    seeds[:, 0] = P[:, 0]
    seeds[:, 1] = P[:, 1]
    # 试验点
    # seeds = radius * np.random.random((50, 2))
    # print(seeds)

    # print("seeds:\n", seeds)
    print("BBox Min:", np.amin(seeds, axis=0),
          "Bbox Max: ", np.amax(seeds, axis=0))

    """
    Compute our Delaunay triangulation of seeds.
    """

    center = np.mean(seeds, axis=0)
    dt = Delaunay2D(center, 50 * radius)
    
    # 插点
    for s in seeds:
        dt.addPoint(s)

    print (len(dt.exportTriangles()), "Delaunay triangles")


    #显示
    import matplotlib.pyplot as plt
    import matplotlib.tri
    import matplotlib.collections

    fig, ax = plt.subplots()
    ax.margins(0.1)
    ax.set_aspect('equal')

    plt.axis([0, 100, -50, 50])
    # plt.axis([-1, radius+1, -1, radius+1])

    # zip() 函数用于将可迭代的对象作为参数，
    # 将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。
    cx, cy = zip(*seeds)
    dt_tris = dt.exportTriangles()
    print(dt_tris)
    # 保存的格式有点问题，是a e+b形式，需要转换一下
    np.savetxt('tri', dt_tris)

    # 蓝色虚线
    ax.triplot(matplotlib.tri.Triangulation(cx, cy, dt_tris), 'bo--')
    # 蓝点加点号
    ax.scatter(seeds[:,0], seeds[:, 1],c='b')
    length = np.arange(len(seeds))
    for i,txt in enumerate(length):
        ax.annotate(txt,(seeds[i]))

    #泰森多边形
    vc, vr = dt.exportVoronoiRegions()
    for r in vr:
        polygon = [vc[i] for i in vr[r]]
        plt.plot(*zip(*polygon), color="red")


    plt.show()
