from manim import *

class GCNExplanation(Scene):
    def construct(self):
        # 第一部分：图结构可视化
        title = Text("图卷积神经网络 (GCN)", font_size=48)
        self.play(Write(title))
        self.wait()
        self.play(FadeOut(title))

        # 创建示例图结构
        nodes = VGroup(*[Circle(radius=0.3, color=BLUE) for _ in range(5)])
        nodes.arrange(RIGHT, buff=1.5)
        edges = VGroup(
            Line(nodes[0].get_right(), nodes[1].get_left(), color=GRAY),
            Line(nodes[1].get_right(), nodes[2].get_left(), color=GRAY),
            Line(nodes[2].get_bottom(), nodes[3].get_top(), color=GRAY),
            Line(nodes[3].get_bottom(), nodes[4].get_top(), color=GRAY),
            Line(nodes[1].get_bottom(), nodes[3].get_left(), color=GRAY)
        )
        graph = VGroup(nodes, edges).scale(0.8).shift(UP*0.5)
        
        self.play(Create(graph))
        self.wait()

        # 添加节点特征标签
        features = VGroup(
            *[MathTex(f"x_{i}").move_to(node) for i, node in enumerate(nodes)]
        )
        self.play(Write(features))
        self.wait()

        # 第二部分：邻接矩阵和特征矩阵
        self.play(graph.animate.shift(LEFT*3).scale(0.6),
                  features.animate.shift(LEFT*3).scale(0.6))

        # 创建邻接矩阵
        adj_matrix = IntegerMatrix(
            [[0, 1, 0, 0, 0],
             [1, 0, 1, 1, 0],
             [0, 1, 0, 1, 0],
             [0, 1, 1, 0, 1],
             [0, 0, 0, 1, 0]],
            v_buff=0.8, h_buff=0.8
        ).scale(0.5).shift(RIGHT*3 + UP*1.5)
        
        adj_title = Text("邻接矩阵 A", font_size=24).next_to(adj_matrix, UP)
        self.play(Write(adj_title), Create(adj_matrix))
        self.wait()

        # 创建特征矩阵
        feat_matrix = DecimalMatrix(
            [[1.2], [0.8], [2.4], [1.6], [3.1]],
            element_to_mobject_config={"num_decimal_places": 1}
        ).scale(0.5).shift(RIGHT*3 + DOWN*1.5)
        
        feat_title = Text("特征矩阵 X", font_size=24).next_to(feat_matrix, UP)
        self.play(Write(feat_title), Create(feat_matrix))
        self.wait()

        # 第三部分：特征传播过程
        arrow = Arrow(adj_matrix.get_left(), feat_matrix.get_right(), color=YELLOW)
        self.play(Create(arrow))
        self.wait()

        # 展示传播公式
        formula = MathTex(
            "H^{(l+1)} = \sigma(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}H^{(l)}W^{(l)})"
        ).scale(0.7).shift(DOWN*2.5)
        self.play(Write(formula))
        self.wait()

        # 第四部分：聚合邻居信息
        self.play(FadeOut(adj_title), FadeOut(feat_title),
                  FadeOut(adj_matrix), FadeOut(feat_matrix),
                  FadeOut(arrow), FadeOut(formula))

        # 高亮中心节点和邻居
        center_node = nodes[2]
        neighbors = VGroup(nodes[1], nodes[3], nodes[4])
        edges_to_center = VGroup(
            edges[1],  # 1->2
            edges[2],  # 2->3
            edges[3]   # 3->4
        )

        self.play(
            Indicate(center_node, color=RED, scale_factor=1.2),
            *[Indicate(n, color=ORANGE, scale_factor=1.1) for n in neighbors],
            edges_to_center.animate.set_color(YELLOW)
        )
        self.wait()

        # 显示聚合过程
        agg_text = Text("聚合邻居特征", font_size=24).next_to(graph, DOWN)
        self.play(Write(agg_text))
        
        # 创建特征聚合动画
        neighbor_feats = VGroup(
            features[1].copy(),
            features[3].copy(),
            features[4].copy()
        )
        self.play(
            *[n.animate.move_to(center_node) for n in neighbor_feats],
            run_time=2
        )
        self.play(FadeOut(neighbor_feats), FadeOut(agg_text))

        # 第五部分：非线性变换
        transform_text = Text("非线性变换", font_size=24).next_to(graph, DOWN)
        self.play(Write(transform_text))
        
        # 显示激活函数
        sigma = MathTex("\sigma").scale(2).next_to(center_node, UP*2)
        self.play(Write(sigma))
        self.play(sigma.animate.shift(DOWN*2), run_time=1.5)
        self.play(FadeOut(sigma), FadeOut(transform_text))

        # 最终输出展示
        output_text = Text("节点特征更新完成！", font_size=28).shift(DOWN*3)
        self.play(Write(output_text))
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)))

# 运行命令：manim -pql gcn.py GCNExplanation