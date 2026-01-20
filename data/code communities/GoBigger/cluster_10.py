# Cluster 10

@pytest.mark.unittest
class TestCollisionDection:

    def test_exhaustive(self):
        border = Border(0, 0, 1000, 1000)
        totol_num = 1000
        query_num = 200
        gallery_list = []
        for i in range(totol_num):
            x = random.randint(border.minx, border.maxx) + random.random()
            y = random.randint(border.miny, border.maxy) + random.random()
            gallery_list.append(BaseBall(i, position=Vector2(x, y), border=border, score=1))
        collision_detection = create_collision_detection('exhaustive', border=border)
        query_list = random.sample(gallery_list, query_num)
        collision_detection.solve(query_list, gallery_list)
        assert True

    def test_precision(self):
        border = Border(0, 0, 1000, 1000)
        totol_num = 1000
        query_num = 200
        gallery_list = []
        for i in range(totol_num):
            x = random.randint(border.minx, border.maxx) + random.random()
            y = random.randint(border.miny, border.maxy) + random.random()
            gallery_list.append(BaseBall(i, position=Vector2(x, y), border=border, score=1))
        collision_detection = create_collision_detection('precision', border=border)
        query_list = random.sample(gallery_list, query_num)
        collision_detection.solve(query_list, gallery_list)
        assert True

    def test_rebuild_quadtree(self):
        border = Border(0, 0, 1000, 1000)
        totol_num = 1000
        query_num = 200
        gallery_list = []
        for i in range(totol_num):
            x = random.randint(border.minx, border.maxx) + random.random()
            y = random.randint(border.miny, border.maxy) + random.random()
            gallery_list.append(BaseBall(i, position=Vector2(x, y), border=border, score=1))
        collision_detection = create_collision_detection('rebuild_quadtree', border=border)
        query_list = random.sample(gallery_list, query_num)
        collision_detection.solve(query_list, gallery_list)
        assert True

    def test_remove_quadtree(self):
        border = Border(0, 0, 1000, 1000)
        totol_num = 1000
        query_num = 200
        change_num = 100
        gallery_list = []
        for i in range(totol_num):
            x = random.randint(border.minx, border.maxx) + random.random()
            y = random.randint(border.miny, border.maxy) + random.random()
            gallery_list.append(BaseBall(i, position=Vector2(x, y), border=border, score=1))
        collision_detection = create_collision_detection('remove_quadtree', border=border)
        collision_detection.solve([], gallery_list)
        change_list = []
        for ball in gallery_list:
            p = random.random()
            if p < change_num / totol_num:
                x = random.randint(border.minx, border.maxx) + random.random()
                y = random.randint(border.miny, border.maxy) + random.random()
                ball.postion = Vector2(x, y)
                change_list.append(ball)
        query_list = random.sample(gallery_list, query_num)
        collision_detection.solve(query_list, change_list)
        assert True

def create_collision_detection(cd_type, **cd_kwargs):
    if cd_type == 'exhaustive':
        return ExhaustiveCollisionDetection(**cd_kwargs)
    if cd_type == 'precision':
        return PrecisionCollisionDetection(**cd_kwargs)
    if cd_type == 'rebuild_quadtree':
        return RebuildQuadTreeCollisionDetection(**cd_kwargs)
    if cd_type == 'remove_quadtree':
        return RemoveQuadTreeCollisionDetection(**cd_kwargs)
    else:
        raise NotImplementedError

class SpeedTest:

    def __init__(self, totol_num, border) -> None:
        self.border = border
        self.totol_num = totol_num
        self.gallery_list = []
        for i in range(totol_num):
            x = random.randint(border.minx, border.maxx) + random.random()
            y = random.randint(border.miny, border.maxy) + random.random()
            self.gallery_list.append(BaseBall(i, position=Vector2(x, y), border=border, score=1))
        self.exhaustive = create_collision_detection('exhaustive', border=border)
        self.precision = create_collision_detection('precision', border=border)
        self.rebuild_quadtree = create_collision_detection('rebuild_quadtree', border=border)
        self.remove_quadtree = create_collision_detection('remove_quadtree', border=border)

    def cal_speed(self, query_num: int, change_num: int, iters: int):
        exhustive_ava_time = 0
        precision_ava_time = 0
        rebuild_tree_ava_time = 0
        remove_tree_ava_time = 0
        self.remove_quadtree.solve([], self.gallery_list)
        for iter in range(iters):
            change_list = []
            for ball in self.gallery_list:
                p = random.random()
                if p < change_num / self.totol_num:
                    x = random.randint(self.border.minx, self.border.maxx) + random.random()
                    y = random.randint(self.border.miny, self.border.maxy) + random.random()
                    ball.postion = Vector2(x, y)
                    change_list.append(ball)
            query_list = random.sample(self.gallery_list, query_num)
            time1 = time.time()
            self.exhaustive.solve(query_list, self.gallery_list)
            time2 = time.time()
            self.precision.solve(query_list, self.gallery_list)
            time3 = time.time()
            self.rebuild_quadtree.solve(query_list, self.gallery_list)
            time4 = time.time()
            self.remove_quadtree.solve(query_list, change_list)
            time5 = time.time()
            exhustive_ava_time += time2 - time1
            precision_ava_time += time3 - time2
            rebuild_tree_ava_time += time4 - time3
            remove_tree_ava_time += time5 - time4
        exhustive_ava_time = int(round(exhustive_ava_time * 1000))
        precision_ava_time = int(round(precision_ava_time * 1000))
        rebuild_tree_ava_time = int(round(rebuild_tree_ava_time * 1000))
        remove_tree_ava_time = int(round(remove_tree_ava_time * 1000))
        return (exhustive_ava_time / iters, precision_ava_time / iters, rebuild_tree_ava_time / iters, remove_tree_ava_time / iters)

