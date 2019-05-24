import numpy as np
from math import sqrt


class Delaunay2D:
    def __init__(self, center=(0, 0), radius=9999):
        """
        center -- 可以用所有点的平均值来取
        radius -- 距离
        """
        center = np.asarray(center)
        # Create coordinates for the corners of the frame
        self.coords = [center+radius*np.array((-1, -1)),
                       center+radius*np.array((+1, -1)),
                       center+radius*np.array((+1, +1)),
                       center+radius*np.array((-1, +1))]

        self.triangles = {}
        self.circles = {}

        # Create two CCW triangles for the frame
        T1 = (0, 1, 3)
        T2 = (2, 3, 1)
        self.triangles[T1] = [T2, None, None]
        self.triangles[T2] = [T1, None, None]

        # 计算三角形外心和半径
        for t in self.triangles:
            self.circles[t] = self.circumcenter(t)

    def circumcenter(self, tri):
        """计算2D三角形的外心和圆周，网址如下：
        http://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html
        """
        pts = np.asarray([self.coords[v] for v in tri])
        # 得到矩阵积
        pts2 = np.dot(pts, pts.T)
        # 矩阵合并
        A = np.bmat([[2 * pts2, [[1],
                                 [1],
                                 [1]]],
                      [[[1, 1, 1, 0]]]])

        # 水平平铺
        b = np.hstack((np.sum(pts * pts, axis=1), [1]))
        x = np.linalg.solve(A, b)
        bary_coords = x[:-1]
        center = np.dot(bary_coords, pts)

        # radius = np.linalg.norm(pts[0] - center) # euclidean distance
        radius = np.sum(np.square(pts[0] - center))  # squared distance
        return (center, radius)

    def inCircleFast(self, tri, p):
        """检查点p是否在tri的预计算外接圆内
        """
        center, radius = self.circles[tri]
        return np.sum(np.square(center - p)) <= radius

    def inCircleRobust(self, tri, p):
        """检查点p是否在tri的预计算外接圆内2.
        速度比1慢
        参考: http://www.cs.cmu.edu/~quake/robust.html
        """
        m1 = np.asarray([self.coords[v] - p for v in tri])
        m2 = np.sum(np.square(m1), axis=1).reshape((3, 1))
        m = np.hstack((m1, m2))    # The 3x3 matrix to check
        return np.linalg.det(m) <= 0

    def addPoint(self, p):
        """
        Bowyer-Watson
        http://en.wikipedia.org/w/index.php?title=Delaunay_triangulation&oldid=626189710
        """
        p = np.asarray(p)
        idx = len(self.coords)
        # 9000个点，需要显示进度
        print("coords[", idx,"] ->",p)
        
        self.coords.append(p)

        # 计算外接圆包括p点的三角（坏三角形）
        bad_triangles = []
        for T in self.triangles:
            # 距离跟半径的比较
            if self.inCircleFast(T, p):
                bad_triangles.append(T)

        # 表现为边和其对应三角。
        boundary = []
        T = bad_triangles[0]
        edge = 0
        while True:
            # 检查三角形T的边缘是否在boundary里...
            tri_op = self.triangles[T][edge]
            if tri_op not in bad_triangles:
                boundary.append((T[(edge+1) % 3], T[(edge-1) % 3], tri_op))
                edge = (edge + 1) % 3
                # 是否遍历完
                if boundary[0][0] == boundary[-1][1]:
                    break
            else:
                # 下一条边
                edge = (self.triangles[tri_op].index(T) + 1) % 3
                T = tri_op

        # delete删除，删出一个空洞
        for T in bad_triangles:
            del self.triangles[T]
            del self.circles[T]

        # 补上删除后的空洞
        new_triangles = []
        # 边（e1，e0）
        for (e0, e1, tri_op) in boundary:
            # 用边缘边和p点创建三角形
            T = (idx, e0, e1)

            # 更新各种附加数据
            # 保存外心和圆周
            self.circles[T] = self.circumcenter(T)

            # 把T的邻接三角设置成tri_op
            self.triangles[T] = [tri_op, None, None]

            # 把tri_op的邻接三角设置成T
            if tri_op:
                for i, neigh in enumerate(self.triangles[tri_op]):
                    if neigh:
                        if e1 in neigh and e0 in neigh:
                            self.triangles[tri_op][i] = T

            # 添加
            new_triangles.append(T)

        # 更新三角新列表
        N = len(new_triangles)
        for i, T in enumerate(new_triangles):
            self.triangles[T][1] = new_triangles[(i+1) % N]   # next
            self.triangles[T][2] = new_triangles[(i-1) % N]   # previous

    def exportTriangles(self):
        """Export the current list of Delaunay triangles
        """
        return [(a-4, b-4, c-4)
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]

    def exportCircles(self):
        """Export the circumcircles as a list of (center, radius)
        """
        return [(self.circles[(a, b, c)][0], sqrt(self.circles[(a, b, c)][1]))
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]

    def exportDT(self):
        """Export the current set of Delaunay coordinates and triangles.
        """
        coord = self.coords[4:]
        tris = [(a-4, b-4, c-4)
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]
        return coord, tris

    def exportExtendedDT(self):
        """Export the Extended Delaunay Triangulation (with the frame vertex).
        """
        return self.coords, list(self.triangles)

    def exportVoronoiRegions(self):
        """Export coordinates and regions of Voronoi diagram as indexed data.
        """

        useVertex = {i: [] for i in range(len(self.coords))}
        vor_coors = []
        index = {}
        for tidx, (a, b, c) in enumerate(sorted(self.triangles)):
            vor_coors.append(self.circles[(a, b, c)][0])
            useVertex[a] += [(b, c, a)]
            useVertex[b] += [(c, a, b)]
            useVertex[c] += [(a, b, c)]

            index[(a, b, c)] = tidx
            index[(c, a, b)] = tidx
            index[(b, c, a)] = tidx

        regions = {}
        for i in range(4, len(self.coords)):
            v = useVertex[i][0][0]
            r = []
            for _ in range(len(useVertex[i])):
                t = [t for t in useVertex[i] if t[0] == v][0]
                r.append(index[t])
                v = t[1]
            regions[i-4] = r
        return vor_coors, regions