from manim import *
import numpy as np

config.media_width = "75%"
config.media_embed = True

class TitleSlide(Scene):
    def construct(self):
        title = Text("Backpropagation", font_size=48)
        subtitle = Text("Understanding Neural Network Training", font_size=32)
        author = Text("Ibrahim Refat, GUB", font_size=24)
        
        subtitle.next_to(title, DOWN)
        author.next_to(subtitle, DOWN, buff=1.0)
        
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle), run_time=1)
        self.play(FadeIn(author), run_time=1)
        self.wait(2)
        
        self.play(FadeOut(title), FadeOut(subtitle), FadeOut(author))
        self.wait(1)

class FourSteps(Scene):
    def construct(self):
        title = Text("Four Steps of Backpropagation", font_size=36)
        self.play(Write(title))
        self.wait(1)
        
        steps = VGroup(
            Text("1. Forward Pass", font_size=28),
            Text("2. Error Calculation", font_size=28),
            Text("3. Backward Pass", font_size=28),
            Text("4. Weight Update", font_size=28)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(title, DOWN, buff=1)
        
        for step in steps:
            self.play(FadeIn(step, shift=RIGHT), run_time=1)
            self.wait(0.5)
        
        self.wait(3)
        self.play(FadeOut(title), *[FadeOut(step) for step in steps])

class NeuralNetwork(VGroup):
    def __init__(self, layer_sizes, neuron_radius=0.15, 
                 layer_spacing=2, neuron_spacing=0.5, **kwargs):
        super().__init__(**kwargs)
        
        self.layer_sizes = layer_sizes
        self.neuron_radius = neuron_radius
        self.layer_spacing = layer_spacing
        self.neuron_spacing = neuron_spacing
        
        self.layers = VGroup()
        self.weights = VGroup()
        
        # Create layers
        for i, size in enumerate(layer_sizes):
            layer = VGroup()
            for j in range(size):
                neuron = Circle(
                    radius=neuron_radius,
                    stroke_color=WHITE,
                    fill_color=BLACK,
                    fill_opacity=1,
                    stroke_width=2
                )
                if i == 0:
                    label = Text(f"X{j+1}", font_size=20).next_to(neuron, DOWN)
                elif i == len(layer_sizes)-1:
                    label = Text("Output", font_size=20).next_to(neuron, DOWN)
                else:
                    label = Text(f"N{i}{j+1}", font_size=20).next_to(neuron, DOWN)
                
                neuron_group = VGroup(neuron, label)
                neuron_group.shift(j * neuron_spacing * DOWN)
                layer.add(neuron_group)
            
            layer.move_to(i * layer_spacing * RIGHT)
            self.layers.add(layer)
        
        # Create weights
        for i in range(len(layer_sizes)-1):
            current_layer = self.layers[i]
            next_layer = self.layers[i+1]
            
            for j, current_neuron in enumerate(current_layer):
                for k, next_neuron in enumerate(next_layer):
                    line = Line(
                        current_neuron[0].get_right(),
                        next_neuron[0].get_left(),
                        stroke_width=1.5,
                        color=BLUE
                    )
                    self.weights.add(line)
        
        self.add(self.layers, self.weights)
    
    def get_neurons(self):
        return [neuron for layer in self.layers for neuron in layer[0]]
    
    def get_weights(self):
        return self.weights

class ForwardPass(Scene):
    def construct(self):
        nn = NeuralNetwork([2, 3, 1])
        nn.scale(0.9).to_edge(UP, buff=1)
        
        title = Text("Forward Pass", font_size=36).to_edge(UP)
        
        self.play(Write(title))
        self.play(Create(nn))
        self.wait(1)
        
        # Animate forward pass
        input_values = [1.5, -0.5]
        activations = []
        
        # Input layer
        for i, neuron in enumerate(nn.layers[0]):
            self.play(
                neuron[0].animate.set_fill(YELLOW, opacity=0.7),
                run_time=0.5
            )
            value = Text(f"{input_values[i]}", font_size=18).move_to(neuron[0])
            self.play(Write(value))
            activations.append((neuron[0], value))
            self.wait(0.5)
        
        # Hidden layer
        hidden_activations = []
        for i, neuron in enumerate(nn.layers[1]):
            # Calculate weighted sum (simplified)
            weighted_sum = sum(input_values) * 0.5  # Simplified calculation
            
            # Draw connections
            for j, input_neuron in enumerate(nn.layers[0]):
                line = nn.weights[i*len(nn.layers[0]) + j].copy()
                line.set_stroke(width=3, color=RED)
                self.play(Create(line), run_time=0.5)
                self.play(FadeOut(line), run_time=0.3)
            
            self.play(
                neuron[0].animate.set_fill(GREEN, opacity=0.7),
                run_time=0.5
            )
            value = Text(f"{weighted_sum:.2f}", font_size=18).move_to(neuron[0])
            self.play(Write(value))
            hidden_activations.append((neuron[0], value))
            self.wait(0.5)
        
        # Output layer
        output_neuron = nn.layers[2][0]
        output_value = sum([float(act[1].text) for act in hidden_activations]) * 0.5
        
        # Draw connections
        for i, hidden_neuron in enumerate(nn.layers[1]):
            line = nn.weights[len(nn.layers[0])*len(nn.layers[1]) + i].copy()
            line.set_stroke(width=3, color=RED)
            self.play(Create(line), run_time=0.5)
            self.play(FadeOut(line), run_time=0.3)
        
        self.play(
            output_neuron[0].animate.set_fill(PURPLE, opacity=0.7),
            run_time=0.5
        )
        output_text = Text(f"{output_value:.2f}", font_size=18).move_to(output_neuron[0])
        self.play(Write(output_text))
        
        # Add loss function box
        loss_box = Rectangle(
            height=1, width=2,
            fill_color=BLUE, fill_opacity=0.2,
            stroke_color=BLUE
        ).next_to(output_neuron, RIGHT, buff=1)
        loss_text = Text("Loss\nFunction", font_size=20).move_to(loss_box)
        arrow = Arrow(output_neuron[0].get_right(), loss_box.get_left(), buff=0.1)
        
        self.play(Create(arrow))
        self.play(Create(loss_box), Write(loss_text))
        
        # Calculate loss (simplified)
        target = 1.0
        loss = (output_value - target)**2
        loss_value = Text(f"Loss: {loss:.4f}", font_size=24).next_to(loss_box, DOWN, buff=0.5)
        self.play(Write(loss_value))
        
        self.wait(3)
        
        # Clean up
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )

class ErrorCalculation(Scene):
    def construct(self):
        title = Text("Error Calculation", font_size=36).to_edge(UP)
        self.play(Write(title))
        
        # Simplified error explanation
        output = Circle(radius=0.5, fill_color=PURPLE, fill_opacity=0.7, stroke_color=WHITE)
        output_label = Text("Output", font_size=20).next_to(output, DOWN)
        output_value = Text("0.75", font_size=24).move_to(output)
        
        target = Circle(radius=0.5, fill_color=GREEN, fill_opacity=0.7, stroke_color=WHITE)
        target_label = Text("Target", font_size=20).next_to(target, DOWN)
        target_value = Text("1.00", font_size=24).move_to(target)
        
        group = VGroup(output, output_label, output_value, target, target_label, target_value)
        group.arrange(RIGHT, buff=2)
        
        self.play(Create(output), Create(target))
        self.play(Write(output_label), Write(target_label))
        self.play(Write(output_value), Write(target_value))
        self.wait(1)
        
        # Error calculation
        error = MathTex("\\text{Error} = \\text{Output} - \\text{Target}").next_to(group, DOWN, buff=1)
        error_calc = MathTex("= 0.75 - 1.00 = -0.25").next_to(error, DOWN)
        
        self.play(Write(error))
        self.play(Write(error_calc))
        self.wait(2)
        
        # Loss function
        loss_title = Text("Loss Function (MSE):", font_size=28).next_to(error_calc, DOWN, buff=1)
        loss_eq = MathTex("L = (\\text{Error})^2 = (-0.25)^2 = 0.0625").next_to(loss_title, DOWN)
        
        self.play(Write(loss_title))
        self.play(Write(loss_eq))
        self.wait(3)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])

class BackwardPass(Scene):
    def construct(self):
        nn = NeuralNetwork([2, 3, 1])
        nn.scale(0.9).to_edge(UP, buff=1)
        
        title = Text("Backward Pass (Backpropagation)", font_size=36).to_edge(UP)
        
        self.play(Write(title))
        self.play(Create(nn))
        self.wait(1)
        
        # Set up network with values
        output_neuron = nn.layers[-1][0]
        output_value = Text("0.75", font_size=18).move_to(output_neuron[0])
        output_neuron[0].set_fill(PURPLE, opacity=0.7)
        
        hidden_neurons = nn.layers[1]
        for i, neuron in enumerate(hidden_neurons):
            neuron[0].set_fill(GREEN, opacity=0.7)
            value = Text(f"0.{i+3}", font_size=18).move_to(neuron[0])
        
        input_neurons = nn.layers[0]
        for i, neuron in enumerate(input_neurons):
            neuron[0].set_fill(YELLOW, opacity=0.7)
            value = Text(f"{1.5 if i==0 else -0.5}", font_size=18).move_to(neuron[0])
        
        self.add(*[neuron[0] for neuron in nn.layers[0]])
        self.add(*[neuron[0] for neuron in nn.layers[1]])
        self.add(output_neuron[0])
        
        # Add loss box
        loss_box = Rectangle(
            height=1, width=2,
            fill_color=BLUE, fill_opacity=0.2,
            stroke_color=BLUE
        ).next_to(output_neuron, RIGHT, buff=1)
        loss_text = Text("Loss: 0.0625", font_size=20).move_to(loss_box)
        arrow = Arrow(output_neuron[0].get_right(), loss_box.get_left(), buff=0.1)
        
        self.play(Create(arrow), Create(loss_box), Write(loss_text))
        self.wait(1)
        
        # Backpropagation animation
        error = MathTex("\\frac{\\partial L}{\\partial \\text{Output}} = 2 \\times \\text{Error} = -0.5").next_to(nn, DOWN)
        self.play(Write(error))
        self.wait(2)
        
        # Show gradient flowing back
        output_to_hidden = VGroup()
        for i in range(len(nn.layers[1])):
            line = nn.weights[len(nn.layers[0])*len(nn.layers[1]) + i].copy()
            line.set_stroke(width=3, color=RED)
            output_to_hidden.add(line)
        
        self.play(LaggedStart(*[Create(line) for line in output_to_hidden], lag_ratio=0.2))
        self.wait(1)
        
        hidden_to_input = VGroup()
        for i in range(len(nn.layers[0])):
            for j in range(len(nn.layers[1])):
                line = nn.weights[i*len(nn.layers[1]) + j].copy()
                line.set_stroke(width=3, color=RED)
                hidden_to_input.add(line)
        
        self.play(LaggedStart(*[Create(line) for line in hidden_to_input], lag_ratio=0.1))
        self.wait(2)
        
        # Show weight updates
        update_text = Text("Updating weights based on gradients...", font_size=24).next_to(error, DOWN)
        self.play(Write(update_text))
        self.wait(2)
        
        # Animate weight changes
        for weight in nn.weights:
            self.play(
                weight.animate.set_stroke(width=2, color=GREEN),
                run_time=0.1
            )
        
        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])

class WeightUpdate(Scene):
    def construct(self):
        title = Text("Weight Update", font_size=36).to_edge(UP)
        self.play(Write(title))
        
        # Create a weight and its components
        weight_line = Line(LEFT, RIGHT, color=BLUE)
        weight_label = MathTex("w_{ij} = 0.5").next_to(weight_line, UP)
        
        neuron_left = Circle(radius=0.3, color=WHITE, fill_opacity=0.7, fill_color=YELLOW)
        neuron_left.move_to(LEFT*2)
        neuron_right = Circle(radius=0.3, color=WHITE, fill_opacity=0.7, fill_color=GREEN)
        neuron_right.move_to(RIGHT*2)
        
        group = VGroup(neuron_left, weight_line, neuron_right, weight_label)
        group.center()
        
        self.play(Create(neuron_left), Create(neuron_right))
        self.play(Create(weight_line), Write(weight_label))
        self.wait(1)
        
        # Show gradient calculation
        gradient = MathTex(
            "\\frac{\\partial L}{\\partial w_{ij}} = \\delta_j \\cdot a_i"
        ).next_to(group, DOWN, buff=1)
        
        self.play(Write(gradient))
        self.wait(2)
        
        # Example values
        delta = MathTex("\\delta_j = -0.2").next_to(gradient, DOWN)
        activation = MathTex("a_i = 0.6").next_to(delta, DOWN)
        
        self.play(Write(delta))
        self.play(Write(activation))
        self.wait(1)
        
        # Calculate gradient
        grad_value = MathTex(
            "\\frac{\\partial L}{\\partial w_{ij}} = -0.2 \\times 0.6 = -0.12"
        ).next_to(activation, DOWN)
        
        self.play(Write(grad_value))
        self.wait(2)
        
        # Weight update
        lr = MathTex("\\text{Learning Rate } \\alpha = 0.1").next_to(grad_value, DOWN)
        update_eq = MathTex(
            "w_{ij}^{\\text{new}} = w_{ij} - \\alpha \\cdot \\frac{\\partial L}{\\partial w_{ij}}"
        ).next_to(lr, DOWN)
        
        update_calc = MathTex(
            "= 0.5 - (0.1 \\times -0.12) = 0.512"
        ).next_to(update_eq, DOWN)
        
        self.play(Write(lr))
        self.play(Write(update_eq))
        self.play(Write(update_calc))
        
        # Animate weight change
        new_label = MathTex("w_{ij} = 0.512").next_to(weight_line, UP)
        self.play(Transform(weight_label, new_label))
        
        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects])

class Conclusion(Scene):
    def construct(self):
        title = Text("Backpropagation Summary", font_size=36).to_edge(UP)
        self.play(Write(title))
        
        bullets = VGroup(
            Text("• Iterative process of forward and backward passes", font_size=24),
            Text("• Calculates gradients using chain rule", font_size=24),
            Text("• Efficiently updates all weights simultaneously", font_size=24),
            Text("• Requires differentiable activation functions", font_size=24),
            Text("• Sensitive to learning rate choice", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(title, DOWN, buff=1)
        
        for bullet in bullets:
            self.play(FadeIn(bullet, shift=RIGHT), run_time=0.7)
            self.wait(0.3)
        
        self.wait(3)
        
        final_text = Text("Thank you!", font_size=42)
        author = Text("Ibrahim Refat, GUB", font_size=28).next_to(final_text, DOWN, buff=0.5)
        
        self.play(
            FadeOut(title),
            *[FadeOut(bullet) for bullet in bullets]
        )
        
        self.play(Write(final_text), Write(author))
        self.wait(3)
        self.play(FadeOut(final_text), FadeOut(author))
