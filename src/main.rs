use ndarray::{array, Array1, ArrayBase, Data, Ix1};
use rand::Rng;

/// 定义单层神经网络结构
struct NeuralNetwork {
    weights: Array1<f32>, // 权重
    bias: f32,            // 偏置
}

impl NeuralNetwork {
    /// 初始化神经网络
    fn new(weights: Array1<f32>, bias: f32) -> Self {
        NeuralNetwork { weights, bias }
    }

    /// 对 Array1<f32> 进行最值归一化
    fn max_normalize<T>(array: &ArrayBase<T, Ix1>) -> Array1<f32>
    where
        T: Data<Elem = f32>,
    {
        // 计算最大值
        let max = array.fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // 如果 max == 0，则返回零数组，避免除以零
        if max < f32::EPSILON {
            return Array1::zeros(array.raw_dim());
        }

        // 归一化
        array.map(|&x| x / max)
    }

    /// Sigmoid 激活函数（带数值稳定性处理）
    fn sigmoid(x: f32) -> f32 {
        let s = 1.0 / (1.0 + (-x).exp());
        // 限制输出范围以避免数值问题
        s.max(1e-8).min(1.0 - 1e-8)
    }

    /// 前向传播计算
    fn forward(&self, input: Array1<f32>) -> f32 {
        let norm_input = Self::max_normalize(&input);
        let z = self.weights.dot(&norm_input) + self.bias;
        Self::sigmoid(z)
    }

    /// 训练神经网络（使用MSE损失函数）
    fn train(&mut self, inputs: &[Array1<f32>], labels: &[f32], learning_rate: f32, epochs: usize) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut total_dw = Array1::zeros(self.weights.raw_dim());
            let mut total_db = 0.0;

            // 遍历每个样本计算梯度
            for (x, &y) in inputs.iter().zip(labels.iter()) {
                // 前向传播
                let x_normalized = Self::max_normalize(x);
                let z = self.weights.dot(&x_normalized) + self.bias;
                let y_hat = Self::sigmoid(z); // y_hat

                // 计算MSE损失（带1/2系数）
                total_loss += 0.5 * (y_hat - y).powi(2);

                // 计算梯度（链式法则）
                let dz = (y_hat - y) * y_hat * (1.0 - y_hat); // 计算损失函数梯度
                total_dw += &(&x_normalized * dz);
                total_db += dz;
            }

            // 平均梯度和损失
            let m = inputs.len() as f32;
            total_dw /= m;
            total_db /= m;
            total_loss /= m;

            // 更新参数
            self.weights -= &(total_dw * learning_rate);
            self.bias -= total_db * learning_rate;

            // 监控训练过程
            if epoch % 100 == 0 {
                println!("Epoch {}: Loss = {:.4}", epoch, total_loss);
            }
        }
    }
}

/// 生成训练数据（v_a, v_b）和标签（1 if v_b > v_a else 0）
fn generate_data(num_samples: usize) -> (Vec<Array1<f32>>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let mut inputs = Vec::with_capacity(num_samples);
    let mut labels = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let v_a = rng.gen_range(0.0..100.0);
        let v_b = rng.gen_range(0.0..100.0);
        inputs.push(array![v_a, v_b]);
        labels.push(if v_b > v_a { 1.0 } else { 0.0 });
    }

    (inputs, labels)
}

fn main() {
    // 生成训练数据
    let (inputs, labels) = generate_data(5000);

    // 随机初始化权重和偏置
    let mut rng = rand::thread_rng();
    let weights = Array1::from(vec![rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)]);
    let bias = rng.gen_range(-1.0..1.0);
    let mut nn = NeuralNetwork::new(weights, bias);

    // 设置训练参数
    let learning_rate = 0.1;
    let epochs = 2000;

    // 开始训练
    nn.train(&inputs, &labels, learning_rate, epochs);

    // 测试训练结果
    let test_cases = vec![
        (array![60.0, 75.0], "追上"),    // v_b > v_a
        (array![6.0, 14.0], "追上"),     // v_b > v_a
        (array![100.0, 50.0], "追不上"), // v_b < v_a
        (array![1.0, 1.0], "追不上"),    // 速度相同
        (array![50.0, 100.0], "追上"),   // v_b > v_a
    ];

    for (input, expected) in test_cases {
        let output = nn.forward(input.clone());
        let prediction = if output > 0.5 { "追上" } else { "追不上" };
        println!(
            "输入: {:?}, 预测: {} ({:.4}), 期望: {}",
            input, prediction, output, expected
        );
    }
}
