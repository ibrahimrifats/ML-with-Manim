from manim import *
import numpy as np
from PIL import Image

config.quality = "medium_quality"
config.frame_size = (1080, 1920)

class MLPTitle(Scene):
    def construct(self):
        title = Text("Multilayer Perceptron (MLP) Deep Dive", font_size=48)
        subtitle = Text("From Neurons to Image Recognition", font_size=36, color=BLUE)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle, shift=UP))
        self.wait(2)
        
        creator = Text("Ibrahim Rifat, GUB", font_size=28)
        creator.to_edge(DOWN)
        
        self.play(FadeIn(creator))
        self.wait(2)
        
        self.play(
            FadeOut(title),
            FadeOut(subtitle),
            FadeOut(creator)
        )

class NeuronImageScene(Scene):
    def construct(self):
        # Neuron diagram
        title = Text("Biological Neuron Inspiration", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Create neuron diagram
        cell_body = Circle(radius=1.0, color=BLUE, fill_opacity=0.3)
        nucleus = Circle(radius=0.3, color=RED, fill_opacity=0.5).move_to(cell_body.get_center())
        
        dendrites = VGroup()
        for i in range(5):
            angle = -60 + i*30
            dendrite = Line(
                cell_body.get_left(),
                cell_body.get_left() + LEFT + np.array([-1, np.sin(angle * DEGREES) * 0.5, 0]),
                stroke_width=3
            )
            dendrites.add(dendrite)
        
        axon = Line(cell_body.get_right(), cell_body.get_right() + RIGHT*2, stroke_width=4)
        axon_terminal = Line(axon.get_end(), axon.get_end() + RIGHT*0.5 + UP*0.5, stroke_width=2)
        
        neuron = VGroup(cell_body, nucleus, dendrites, axon, axon_terminal)
        neuron.scale(0.7).shift(DOWN*0.5)
        
        self.play(Create(neuron))
        self.wait(1)
        
        # Labels
        dendrite_label = Text("Dendrites (Inputs)", font_size=24).next_to(dendrites, LEFT)
        axon_label = Text("Axon (Output)", font_size=24).next_to(axon, RIGHT)
        cell_label = Text("Cell Body\n(Processing)", font_size=24).next_to(cell_body, UP)
        
        self.play(Write(dendrite_label), run_time=1)
        self.play(Write(axon_label), run_time=1)
        self.play(Write(cell_label), run_time=1)
        self.wait(2)
        
        # Artificial neuron
        art_title = Text("Artificial Neuron Model", font_size=36, color=YELLOW)
        art_title.to_edge(UP)
        
        self.play(
            ReplacementTransform(title, art_title),
            FadeOut(dendrite_label),
            FadeOut(axon_label),
            FadeOut(cell_label)
        )
        
        # Create artificial neuron
        inputs = VGroup()
        for i in range(3):
            input_dot = Dot(LEFT*3 + UP*(1-i), color=GREEN)
            input_label = MathTex(f"x_{i+1}", font_size=24).next_to(input_dot, LEFT)
            inputs.add(VGroup(input_dot, input_label))
        
        neuron_circle = Circle(radius=0.8, color=RED).shift(RIGHT)
        neuron_label = Text("Σ → f", font_size=28).move_to(neuron_circle)
        
        output = Dot(RIGHT*3, color=BLUE)
        output_label = MathTex("y", font_size=28).next_to(output, RIGHT)
        
        weights = VGroup()
        for i, inp in enumerate(inputs):
            weight = Line(inp[0].get_right(), neuron_circle.get_left(), stroke_width=2)
            weight_label = MathTex(f"w_{i+1}", font_size=20).move_to(weight).shift(UP*0.2)
            weights.add(VGroup(weight, weight_label))
        
        output_line = Line(neuron_circle.get_right(), output.get_left(), stroke_width=3)
        
        self.play(
            FadeOut(neuron),
            Create(inputs),
            Create(neuron_circle),
            Create(neuron_label),
            Create(weights),
            Create(output_line),
            Create(output),
            Create(output_label)
        )
        self.wait(2)
        
        # Math explanation
        math_tex = MathTex(
            "y = f\\left(\\sum_{i=1}^{n} w_i x_i + b\\right)",
            font_size=40
        ).to_edge(DOWN)
        
        self.play(Write(math_tex))
        self.wait(3)

class MLPAnimation(Scene):
    def construct(self):
        # Scene 1: Create 5x5 L image
        title = Text("MLP for Image Recognition", font_size=40)
        title.to_edge(UP, buff=0.1)  # 10px from top
        self.play(Write(title))
        
        pixel_grid = VGroup()
        pixel_values = np.zeros((5, 5))
        
        # Create L pattern
        for i in range(5):
            pixel_values[i][0] = 1
        for j in range(3):
            pixel_values[4][j] = 1
        
        for i in range(5):
            row = VGroup()
            for j in range(5):
                color = WHITE if pixel_values[i][j] == 1 else BLACK
                square = Square(side_length=0.8, stroke_width=2, fill_color=color, fill_opacity=1)
                square.move_to(np.array([j, -i, 0]) * 0.8)
                row.add(square)
            pixel_grid.add(row)
        
        pixel_grid.center().shift(UP*0.5)
        
        grid_label = Text("5x5 Input Image (Letter 'L')", font_size=28).next_to(pixel_grid, DOWN, buff=0.3)
        
        self.play(Create(pixel_grid), Write(grid_label))
        self.wait(2)
        
        # Clear screen
        self.play(FadeOut(title), FadeOut(pixel_grid), FadeOut(grid_label))
        self.wait(0.5)
        
        # Scene 2: Flatten image
        flat_title = Text("Flatten to Vector", font_size=40)
        flat_title.to_edge(UP, buff=0.1)
        self.play(Write(flat_title))
        
        input_vector = VGroup()
        for i in range(25):
            pixel = Square(side_length=0.4, fill_color=interpolate_color(BLACK, WHITE, pixel_values.flatten()[i]), 
                           fill_opacity=1, stroke_width=1)
            input_vector.add(pixel)
        
        input_vector.arrange(RIGHT, buff=0.05).center().shift(DOWN*0.5)
        
        vector_label = Text("25-dimensional input vector", font_size=28).next_to(input_vector, DOWN, buff=0.3)
        
        self.play(Create(input_vector), Write(vector_label))
        self.wait(2)
        
        # Clear screen
        self.play(FadeOut(flat_title), FadeOut(vector_label))
        self.wait(0.5)
        
        # Scene 3: MLP architecture - Transform vector to neurons
        mlp_title = Text("Multilayer Perceptron Architecture", font_size=36, color=YELLOW)
        mlp_title.to_edge(UP, buff=0.1)
        self.play(Write(mlp_title))
        
        # Create layers
        layers = VGroup()
        layer_labels = VGroup()
        layer_sizes = [25, 16, 8, 2]  # Input, Hidden1, Hidden2, Output
        layer_positions = [-4, -1, 2, 5]
        
        for i, (size, x) in enumerate(zip(layer_sizes, layer_positions)):
            layer = VGroup()
            for j in range(min(8, size)):  # Show max 8 neurons for display
                neuron = Circle(radius=0.2, color=BLUE, fill_opacity=0.3)
                neuron.move_to(np.array([x, (min(8, size)/2 - j - 0.5)*0.6, 0]))
                layer.add(neuron)
            
            # Add dots if layer has more than 8 neurons
            if size > 8:
                dots = VGroup()
                for k in range(3):
                    dot = Dot(color=BLUE, radius=0.05)
                    dot.move_to(np.array([x, -2.5 - k*0.2, 0]))
                    dots.add(dot)
                layer.add(dots)
            
            layers.add(layer)
            
            if i == 0:
                label = Text("Input\nLayer", font_size=20)
            elif i == len(layer_sizes)-1:
                label = Text("Output\nLayer", font_size=20)
            else:
                label = Text(f"Hidden\nLayer {i}", font_size=20)
            
            label.next_to(layer, DOWN, buff=0.3)
            layer_labels.add(label)
        
        layers.center().shift(DOWN*0.5)
        layer_labels.center().shift(DOWN*2.8)
        
        # Transform input vector to first layer neurons
        self.play(ReplacementTransform(input_vector, layers[0]))
        self.play(Create(layers[1:]))  # Create other layers
        self.play(Write(layer_labels))
        self.wait(1)
        
        # Now create connections AFTER neurons are shown
        connections = VGroup()
        for i in range(len(layer_sizes)-1):
            layer1_size = min(8, layer_sizes[i])
            layer2_size = min(8, layer_sizes[i+1])
            
            for j in range(layer1_size):
                for k in range(layer2_size):
                    line = Line(
                        layers[i][j].get_right(),
                        layers[i+1][k].get_left(),
                        stroke_width=0.8,
                        stroke_opacity=0.4,
                        color=GRAY
                    )
                    connections.add(line)
        
        self.play(Create(connections), run_time=2)
        self.wait(2)
        
        # Clear screen
        self.play(FadeOut(mlp_title), FadeOut(layers), FadeOut(layer_labels), FadeOut(connections))
        self.wait(0.5)
        
        # Scene 4: Forward propagation
        fp_title = Text("Forward Propagation", font_size=36, color=GREEN)
        fp_title.to_edge(UP, buff=0.1)
        self.play(Write(fp_title))
        
        # Recreate simplified network for propagation demo
        simple_layers = VGroup()
        simple_connections = VGroup()
        layer_positions = [-3, 0, 3]
        layer_sizes = [5, 3, 2]
        
        for i, (size, x) in enumerate(zip(layer_sizes, layer_positions)):
            layer = VGroup()
            for j in range(size):
                neuron = Circle(radius=0.3, color=BLUE, fill_opacity=0.3, stroke_width=2)
                neuron.move_to(np.array([x, (size/2 - j - 0.5)*1.2, 0]))
                layer.add(neuron)
            simple_layers.add(layer)
        
        # Create connections
        for i in range(len(layer_sizes)-1):
            for j in range(layer_sizes[i]):
                for k in range(layer_sizes[i+1]):
                    line = Line(
                        simple_layers[i][j].get_right(),
                        simple_layers[i+1][k].get_left(),
                        stroke_width=2,
                        stroke_opacity=0.6,
                        color=GRAY
                    )
                    simple_connections.add(line)
        
        self.play(Create(simple_layers), Create(simple_connections))
        
        # Animate data flow
        for layer_idx in range(len(simple_layers)):
            pulses = VGroup()
            for neuron in simple_layers[layer_idx]:
                pulse = neuron.copy().set(color=YELLOW, fill_opacity=0.9)
                pulses.add(pulse)
            
            self.play(
                LaggedStart(*[Flash(neuron, color=YELLOW, flash_radius=0.4) 
                            for neuron in simple_layers[layer_idx]]),
                run_time=1.5
            )
            self.wait(0.5)
        
        self.wait(2)
        
        # Clear screen
        self.play(FadeOut(fp_title), FadeOut(simple_layers), FadeOut(simple_connections))
        self.wait(0.5)
        
        # Scene 5: Activation functions
        act_title = Text("Activation Functions", font_size=36, color=ORANGE)
        act_title.to_edge(UP, buff=0.1)
        self.play(Write(act_title))
        
        # ReLU visualization - positioned more centrally
        relu_axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[0, 2, 0.5],
            axis_config={"color": BLUE, "include_numbers": True},
            x_length=4,
            y_length=2.5
        ).shift(LEFT*3 + DOWN*0.3)
        
        relu_label = Text("ReLU: max(0, x)", font_size=20, color=YELLOW).next_to(relu_axes, UP, buff=0.1)
        relu_formula = MathTex(r"f(x) = \max(0, x)", font_size=18, color=WHITE).next_to(relu_label, DOWN, buff=0.1)
        
        relu_graph = relu_axes.plot(
            lambda x: max(0, x),
            color=YELLOW,
            stroke_width=3,
            x_range=[-2, 2]
        )
        
        # Show example calculation
        example_input = Text("Input: [-1.2, 0.8, -0.3]", font_size=16, color=WHITE)
        example_output = Text("ReLU Output: [0, 0.8, 0]", font_size=16, color=YELLOW)
        example_group = VGroup(example_input, example_output).arrange(DOWN, buff=0.2)
        example_group.next_to(relu_axes, DOWN, buff=0.3)
        
        # Softmax visualization - positioned to avoid overflow
        softmax_section = VGroup()
        softmax_title = Text("Softmax for Classification", font_size=20, color=PURPLE)
        
        # Create probability bars for L vs Not L
        classes = ["Letter 'L'", "Not 'L'"]
        probabilities = [0.89, 0.11]  # More realistic for good training
        colors = [GREEN, RED]
        
        bars_group = VGroup()
        for i, (cls, prob, color) in enumerate(zip(classes, probabilities, colors)):
            # Create bar
            bar = Rectangle(
                width=2.5,
                height=prob * 2,
                fill_color=color,
                fill_opacity=0.7,
                stroke_width=2,
                stroke_color=WHITE
            )
            
            # Position bars
            bar.move_to(RIGHT*2.5 + DOWN*(1 - prob))
            
            # Labels
            class_label = Text(cls, font_size=14, color=WHITE)
            prob_label = Text(f"{prob:.2f}", font_size=14, color=WHITE)
            
            class_label.next_to(bar, DOWN, buff=0.1)
            prob_label.next_to(bar, UP, buff=0.1)
            
            bar_group = VGroup(bar, class_label, prob_label)
            bars_group.add(bar_group)
            
            if i == 0:  # Position for first bar
                bar.shift(LEFT*0.8)
                class_label.shift(LEFT*0.8)
                prob_label.shift(LEFT*0.8)
            else:  # Position for second bar
                bar.shift(RIGHT*0.8)
                class_label.shift(RIGHT*0.8)
                prob_label.shift(RIGHT*0.8)
        
        softmax_formula = MathTex(r"\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}", 
                                 font_size=16, color=WHITE)
        
        softmax_section.add(softmax_title, bars_group, softmax_formula)
        softmax_section.arrange(DOWN, buff=0.3)
        softmax_section.shift(RIGHT*2.5 + DOWN*0.3)
        
        # Animate ReLU first
        self.play(Create(relu_axes), Write(relu_label))
        self.play(Create(relu_graph), Write(relu_formula))
        self.play(Write(example_group))
        self.wait(2)
        
        # Then Softmax
        self.play(Write(softmax_title), Write(softmax_formula))
        self.play(Create(bars_group))
        
        # Highlight the prediction
        prediction = Text("Prediction: Letter 'L' (89% confidence)", 
                         font_size=18, color=GREEN, weight=BOLD)
        prediction.to_edge(DOWN, buff=0.3)
        self.play(Write(prediction))
        self.wait(3)
        
        # Clear screen
        self.play(
            FadeOut(act_title), FadeOut(relu_axes), FadeOut(relu_label), FadeOut(relu_graph),
            FadeOut(relu_formula), FadeOut(example_group), FadeOut(softmax_section), FadeOut(prediction)
        )
        self.wait(0.5)
        
        # Scene 6: Training process - Realistic for L character detection
        train_title = Text("Training: Learning to Recognize 'L'", font_size=36, color=PURPLE)
        train_title.to_edge(UP, buff=0.1)
        self.play(Write(train_title))
        
        # Show training data examples
        training_examples = VGroup()
        
        # Create mini 3x3 representations of training data
        patterns = [
            # L pattern
            [[1, 0, 0], [1, 0, 0], [1, 1, 1]],
            # Not L patterns
            [[1, 1, 1], [0, 1, 0], [0, 1, 0]],  # T
            [[1, 1, 1], [1, 0, 1], [1, 1, 1]],  # O-like
        ]
        labels = ["L", "T", "O"]
        
        for i, (pattern, label) in enumerate(zip(patterns, labels)):
            example_group = VGroup()
            
            # Create mini grid
            grid = VGroup()
            for row in range(3):
                for col in range(3):
                    color = WHITE if pattern[row][col] == 1 else BLACK
                    square = Square(side_length=0.2, fill_color=color, fill_opacity=1, stroke_width=1)
                    square.move_to([col*0.2, -row*0.2, 0])
                    grid.add(square)
            
            # Label
            label_text = Text(label, font_size=16, color=YELLOW if label == "L" else RED)
            label_text.next_to(grid, DOWN, buff=0.1)
            
            example_group.add(grid, label_text)
            example_group.move_to([-3 + i*2, 1.5, 0])
            training_examples.add(example_group)
        
        training_label = Text("Training Examples", font_size=20, color=WHITE)
        training_label.next_to(training_examples, UP, buff=0.2)
        
        self.play(Write(training_label))
        self.play(Create(training_examples))
        self.wait(2)
        
        # Create simplified network for L detection (25 → 10 → 5 → 2)
        network_layers = VGroup()
        layer_info = [
            (25, -4, "Input\n(25 pixels)"),
            (10, -1.5, "Hidden 1\n(10 neurons)"),
            (5, 1, "Hidden 2\n(5 neurons)"),
            (2, 3.5, "Output\n(L vs Not-L)")
        ]
        
        for size, x_pos, label in layer_info:
            layer = VGroup()
            display_size = min(6, size)  # Show max 6 neurons
            
            for j in range(display_size):
                neuron = Circle(radius=0.15, color=BLUE, fill_opacity=0.4, stroke_width=1)
                neuron.move_to([x_pos, (display_size/2 - j - 0.5)*0.4, 0])
                layer.add(neuron)
            
            # Add dots for larger layers
            if size > 6:
                dots = Text("⋮", font_size=20, color=BLUE)
                dots.move_to([x_pos, -1.5, 0])
                layer.add(dots)
            
            layer_label = Text(label, font_size=12, color=WHITE)
            layer_label.next_to(layer, DOWN, buff=0.2)
            
            network_layers.add(VGroup(layer, layer_label))
        
        network_layers.shift(DOWN*0.8)
        
        self.play(Create(network_layers))
        self.wait(1)
        
        # Show forward pass with our L example
        forward_text = Text("Forward Pass: Predicting 'L'", font_size=20, color=GREEN)
        forward_text.next_to(network_layers, DOWN, buff=0.5)
        self.play(Write(forward_text))
        
        # Animate forward propagation
        for i in range(len(layer_info)):
            layer = network_layers[i][0]  # Get the neuron layer
            for neuron in layer:
                if hasattr(neuron, 'radius'):  # Skip dots
                    self.play(Flash(neuron, color=YELLOW, flash_radius=0.2), run_time=0.1)
            self.wait(0.3)
        
        # Show prediction result
        prediction_result = Text("Prediction: 'L' - 92% confidence", font_size=18, color=GREEN)
        actual_label = Text("Actual: 'L' ✓", font_size=18, color=GREEN)
        result_group = VGroup(prediction_result, actual_label).arrange(DOWN, buff=0.1)
        result_group.next_to(forward_text, DOWN, buff=0.3)
        
        self.play(Write(result_group))
        self.wait(2)
        
        # Show error calculation and backpropagation
        error_calc = Text("Error = |Predicted - Actual| = 0.08", font_size=16, color=ORANGE)
        backprop_text = Text("Backpropagation: Adjusting weights to reduce error", font_size=16, color=RED)
        learning_group = VGroup(error_calc, backprop_text).arrange(DOWN, buff=0.2)
        learning_group.next_to(result_group, DOWN, buff=0.4)
        
        self.play(Write(learning_group))
        
        # Animate gradient flow (simplified)
        gradient_indicators = VGroup()
        for i in range(len(layer_info)-1):
            arrow = Arrow(
                network_layers[i+1][0].get_left() + LEFT*0.2,
                network_layers[i][0].get_right() + RIGHT*0.2,
                color=RED,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.1
            )
            gradient_indicators.add(arrow)
        
        self.play(Create(gradient_indicators))
        
        # Show weight update
        update_text = Text("Weights Updated! Model becomes more accurate.", 
                          font_size=16, color=YELLOW)
        update_text.next_to(learning_group, DOWN, buff=0.3)
        self.play(Write(update_text))
        self.wait(2)
        
        # Final summary
        summary_text = Text("After many training examples, the model learns\nto distinguish 'L' from other characters!", 
                          font_size=18, color=GREEN)
        summary_text.to_edge(DOWN, buff=0.2)
        self.play(Write(summary_text))
        self.wait(3)
        
        # Clear everything
        self.play(
            FadeOut(train_title), FadeOut(training_label), FadeOut(training_examples),
            FadeOut(network_layers), FadeOut(forward_text), FadeOut(result_group),
            FadeOut(learning_group), FadeOut(gradient_indicators), FadeOut(update_text),
            FadeOut(summary_text)
        )
        self.wait(1)


class FinalSummary(Scene):
    def construct(self):
        # Key components
        components = VGroup(
            Text("MLP Key Components:", font_size=40, color=YELLOW),
            Text("- Input Layer (Flattened Image)", font_size=32),
            Text("- Hidden Layers (Feature Extraction)", font_size=32),
            Text("- Activation Functions (ReLU, Softmax)", font_size=32),
            Text("- Weights & Biases (Learned Parameters)", font_size=32),
            Text("- Backpropagation (Training Process)", font_size=32)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        
        components.center()
        
        self.play(LaggedStart(*[FadeIn(comp, shift=UP) for comp in components], run_time=3))
        self.wait(3)
        
        # Applications
        apps_title = Text("Real-World Applications:", font_size=40, color=GREEN)
        apps_title.to_edge(UP)
        
        applications = VGroup(
            Text("• Image Recognition", font_size=32),
            Text("• Natural Language Processing", font_size=32),
            Text("• Financial Forecasting", font_size=32),
            Text("• Medical Diagnosis", font_size=32)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.7).next_to(apps_title, DOWN, buff=1)
        
        self.play(
            FadeOut(components),
            FadeIn(apps_title, shift=DOWN)
        )
        self.play(LaggedStart(*[FadeIn(app, shift=LEFT) for app in applications], run_time=2))
        self.wait(3)
        
        # Final animation
        mlp_diagram = self.create_mlp_diagram()
        mlp_diagram.scale(0.7).to_edge(DOWN)
        
        self.play(
            FadeOut(apps_title),
            FadeOut(applications),
            FadeIn(mlp_diagram, shift=UP)
        )
        
        final_text = Text("Mastering Multilayer Perceptrons", font_size=48, color=BLUE)
        self.play(Write(final_text))
        self.wait(3)
        
    def create_mlp_diagram(self):
        layers = VGroup()
        layer_sizes = [4, 6, 4, 2]  # Simplified for summary
        layer_positions = [-4, -1, 2, 5]
        
        for i, (size, x) in enumerate(zip(layer_sizes, layer_positions)):
            layer = VGroup()
            for j in range(size):
                neuron = Circle(radius=0.15, color=BLUE, fill_opacity=0.3)
                neuron.move_to(np.array([x, (size/2 - j - 0.5)*0.6, 0]))
                layer.add(neuron)
            layers.add(layer)
            
            if i == 0:
                label = Text("Input", font_size=20)
            elif i == len(layer_sizes)-1:
                label = Text("Output", font_size=20)
            else:
                label = Text(f"Hidden {i}", font_size=20)
            
            label.next_to(layer, DOWN, buff=0.2)
            layers.add(label)
        
        # Connections
        connections = VGroup()
        for i in range(len(layer_sizes)-1):
            for j in range(layer_sizes[i]):
                for k in range(layer_sizes[i+1]):
                    line = Line(
                        layers[2*i][j].get_right(),
                        layers[2*(i+1)][k].get_left(),
                        stroke_width=0.8,
                        stroke_opacity=0.2
                    )
                    connections.add(line)
        
        return VGroup(connections, layers)
