use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use rand::Rng;

// -------------------
// 核心Trait定义
// -------------------

/// 神经网络层统一接口
trait Layer {
    /// 前向传播
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64>;

    /// 反向传播（返回输入梯度）
    fn backward(&mut self, grad_output: &Array2<f64>) -> Array2<f64>;

    /// 参数更新（默认空实现）
    fn apply_update(&mut self, _optimizer: &dyn Optimizer) {}
}

/// 优化器统一接口
trait Optimizer {
    /// 计算参数更新量（支持任意维度）
    fn compute_update(&self, param: &Array2<f64>, grad: &Array2<f64>) -> Array2<f64>;
}

/// 激活函数接口
trait Activation {
    fn forward(&self, input: &Array2<f64>) -> Array2<f64>;
    fn backward(&self, input: &Array2<f64>, grad_output: &Array2<f64>) -> Array2<f64>;
}

/// 损失函数接口
trait Loss {
    fn compute(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64;
    fn gradient(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64>;
}

// -------------------
// 全连接层实现
// -------------------

struct LinearLayer {
    weights: Array2<f64>,
    biases: Array1<f64>,
    weight_grad: Option<Array2<f64>>,
    bias_grad: Option<Array1<f64>>,
    input_cache: Option<Array2<f64>>
}

impl LinearLayer {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        let shape = (input_dim, output_dim);
        let weights = Array2::random(shape, StandardNormal) * 0.01;
        let biases = Array1::zeros(output_dim);

        LinearLayer {
            weights,
            biases,
            weight_grad: None,
            bias_grad: None,
            input_cache: None
        }
    }
}

impl Layer for LinearLayer {
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        self.input_cache = Some(input.clone());
        input.dot(&self.weights) + &self.biases
    }

    fn backward(&mut self, grad_output: &Array2<f64>) -> Array2<f64> {
        // 计算权重和偏置的梯度
        let input = self.input_cache.as_ref().expect("No input cache found");
        self.weight_grad = Some(input.t().dot(grad_output));
        self.bias_grad = Some(grad_output.sum_axis(Axis(0)));
        // 返回传递给前一层的梯度
        grad_output.dot(&self.weights.t())
    }

    fn apply_update(&mut self, optimizer: &dyn Optimizer) {
        if let (Some(w_grad), Some(b_grad)) = (&self.weight_grad, &self.bias_grad) {
            // 为了处理偏置更新，将一维转换为二维
            let b_grad_2d = b_grad.clone().insert_axis(Axis(0));
            let b_param_2d = self.biases.clone().insert_axis(Axis(0));

            // 计算更新量
            let delta_w = optimizer.compute_update(&self.weights, w_grad);
            let delta_b = optimizer.compute_update(&b_param_2d, &b_grad_2d);

            // 应用更新
            self.weights -= &delta_w;
            self.biases -= &delta_b.index_axis(Axis(0), 0);
        }

        // 清空梯度缓存
        self.weight_grad = None;
        self.bias_grad = None;
    }
}

// -------------------
// 激活层实现（示例）
// -------------------

struct ReLU;

impl Activation for ReLU {
    fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        input.mapv(|x| x.max(0.0))
    }

    fn backward(&self, input: &Array2<f64>, grad_output: &Array2<f64>) -> Array2<f64> {
        grad_output * input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}

struct Sigmoid;

impl Activation for Sigmoid {
    fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn backward(&self, input: &Array2<f64>, grad_output: &Array2<f64>) -> Array2<f64> {
        let sig = self.forward(input);
        grad_output * &sig * &(1.0 - &sig)
    }
}

struct ActivationFn {
    activation: Box<dyn Activation>,
    input_cache: Option<Array2<f64>>,
}

impl ActivationFn {
    fn new(activation: Box<dyn Activation>)->Self {
        ActivationFn {
            activation:activation,
            input_cache: None,
        }
    }
}

impl Layer for ActivationFn {
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        self.input_cache = Some(input.clone());
        self.activation.forward(input)
    }

    fn backward(&mut self, grad_output: &Array2<f64>) -> Array2<f64> {
        let input = self.input_cache.as_ref().expect("No input cache found");
        self.activation.backward(input, grad_output)
    }
    
    // 无需实现apply_update，使用默认空实现
}


// -------------------
// 损失函数实现（MSE）
// -------------------

struct MSE;

impl Loss for MSE {
    fn compute(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        let diff = predictions - targets;
        diff.mapv(|x| x.powi(2)).mean().unwrap()
    }

    fn gradient(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
        (predictions - targets) * 2.0 / predictions.shape()[0] as f64
    }
}

// -------------------
// 优化器实现（SGD）
// -------------------

struct SGD {
    learning_rate: f64,
}

impl SGD {
    fn new(learning_rate: f64) -> Self {
        SGD { learning_rate }
    }
}

impl Optimizer for SGD {
    fn compute_update(&self, _param: &Array2<f64>, grad: &Array2<f64>) -> Array2<f64> {
        grad * self.learning_rate
    }
}

// -------------------
// 神经网络模型
// -------------------

struct Model {
    layers: Vec<Box<dyn Layer>>,
    loss: Box<dyn Loss>,
    optimizer: Box<dyn Optimizer>,
}

impl Model {
    fn new(
        layers: Vec<Box<dyn Layer>>,
        loss: Box<dyn Loss>,
        optimizer: Box<dyn Optimizer>,
    ) -> Self {
        Model {
            layers,
            loss,
            optimizer,
        }
    }

    pub fn train(&mut self, inputs: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        // 前向传播
        let mut output = inputs.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }

        // 计算损失梯度
        let loss_grad = self.loss.gradient(&output, targets);

        // 反向传播
        let mut grad = loss_grad;
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }

        // 参数更新
        for layer in &mut self.layers {
            layer.apply_update(&*self.optimizer);
        }

        self.loss.compute(&output, targets)
    }

    pub fn predict(&mut self, inputs: &Array2<f64>)->Array2<f64>{
        let mut output = inputs.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }
}

// -------------------
// 使用示例
// -------------------
/// 生成 XOR 数据
/// 返回值为 (inputs, labels)
fn generate_xor_data(num_samples: usize) -> (Array2<f64>, Array2<f64>) {
    let mut inputs = Array2::<f64>::zeros((num_samples, 2));
    let mut labels = Array2::<f64>::zeros((num_samples, 1));
    let mut rng = rand::thread_rng();

    for i in 0..num_samples {
        // 随机生成 0 或 1
        let x1 = rng.gen_range(0..=1) as f64;
        let x2 = rng.gen_range(0..=1) as f64;
        inputs[[i, 0]] = x1;
        inputs[[i, 1]] = x2;
        
        // XOR 运算：如果两个输入不同，则结果为 1，否则为 0
        labels[[i, 0]] = if x1 != x2 { 1.0 } else { 0.0 };
    }
    
    (inputs, labels)
}

fn main() {
    // 创建网络结构
    let mut model = Model::new(
        vec![
            Box::new(LinearLayer::new(2, 8)),
            Box::new(ActivationFn::new(Box::new(ReLU))),
            Box::new(LinearLayer::new(8, 1)),
            Box::new(ActivationFn::new(Box::new(Sigmoid))),
        ],
        Box::new(MSE),
        Box::new(SGD::new(0.05)),
    );

    for epoch in 0..80000 {
        // 训练数据（XOR问题）
        let (inputs, labels) = generate_xor_data(20);
        // 训练模型
        let loss_value = model.train(&inputs, &labels);
        // 打印训练进度
        if epoch % 1000 == 0 {
            println!("Epoch {} | Loss: {:.8}", epoch, loss_value);
        }
    }
    // 测试预测
    let (test_input,labels) = generate_xor_data(10);
    let output = model.predict(&test_input);
    println!("\nFinal Predictions:");
    for i in 0..10 {
        println!("Input [{:.1}, {:.1}] -> {:.4}, target: {:.1}", test_input[[i,0]],test_input[[i,1]],output[[i, 0]],labels[[i,0]]);
    }
   
}
