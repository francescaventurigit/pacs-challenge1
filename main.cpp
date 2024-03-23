#include <iostream>
#include <memory>
#include <Eigen/Dense>
#include <functional>
#include "gradient_descent.hpp"


int main() {

    // Define the function f ...
    std::function<double(const Eigen::VectorXd&)> f = [] (const Eigen::VectorXd& x){
        return x[0]*x[1] + 4*x[0]*x[0]*x[0]*x[0] + x[1]*x[1] + 3*x[0];
        };

    // Set up the stepsize for finite differences computation of the gradient
    double h = 1e-3;

    // ... and define the gradient (centered finite differences):
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> grad_f = [h, &f] (const Eigen::VectorXd& x) {
        Eigen::VectorXd grad(x.size());
        for (int i = 0; i < x.size(); ++i) {
            Eigen::VectorXd x_plus_h = x;
            Eigen::VectorXd x_minus_h = x;
            x_plus_h[i] += h;
            x_minus_h[i] -= h;
            grad[i] = (f(x_plus_h) - f(x_minus_h)) / (2 * h);
        }
        return grad;
    };

    // Set up parameters
    std::cout << "Setting up parameters" << std::endl;
    
    OptParams opt_params;
    opt_params.initial_guess = Eigen::VectorXd::Zero(2);
    opt_params.epsilon_r = 1e-6;
    opt_params.epsilon_s = 1e-6;
    opt_params.max_iterations = 1000;
    
    DecayParams decay_params;
    decay_params.alpha0 = 0.1;
    decay_params.sigma = 0.4;
    decay_params.mu = 1.1;

    // Let the user choose the Learning Rate Decay Strategy
    std::cout << "\nChoose the Learning Rate Strategy: 'constant', 'exponential', 'inverse', 'armijo'"
              << "\nPress enter to use the default ('armijo')" << std::endl;
    
    std::string lr_decay_str;
    std::getline(std::cin, lr_decay_str);

    if (lr_decay_str.empty()) {
        lr_decay_str = "armijo";
    }

    std::shared_ptr<LearningRateDecay> decay = nullptr;

    if (lr_decay_str == "constant") {
        decay = std::make_shared<NoDecay>(decay_params);
    } else if (lr_decay_str == "exponential") {
        decay = std::make_shared<ExponentialDecay>(decay_params);
    } else if (lr_decay_str == "inverse") {
        decay = std::make_shared<InverseDecay>(decay_params);
    } else if (lr_decay_str == "armijo") {
        decay = std::make_shared<ArmijoDecay>(decay_params, f, grad_f);
    } else {
        std::cerr << "Invalid learning rate decay strategy. Check for typos!" << std::endl;
        return 1;
    }

    // let the user choose the Optimization Method
    std::cout << "\nChoose the Optimization Method: 'Standard GD', 'Momentum', 'Nesterov'"
              << "\nPress enter to use the default ('Standard GD') \n" << std::endl;
    
    std::string opt_str;
    std::getline(std::cin, opt_str);

    if (opt_str.empty()) {
        opt_str = "Standard GD";
    }

    std::shared_ptr<Optimizer> optimizer = nullptr;

    if (opt_str == "Standard GD") {
        optimizer = std::make_shared<GradientDescent>(f, grad_f, opt_params, decay);
    } else if (opt_str == "Momentum") {
        optimizer = std::make_shared<Momentum>(f, grad_f, opt_params, decay);;
    } else if (opt_str == "Nesterov") {
        optimizer = std::make_shared<Nesterov>(f, grad_f, opt_params, decay);;
    
    } else {
        std::cerr << "Invalid Optimization Method. Check for typos!" << std::endl;
        return 1;
    }

    // Perform gradient descent
    Eigen::VectorXd minimum = optimizer->minimize();

    // Output the result
    std::cout << "Minimum found at: (" << minimum[0] << ", " << minimum[1] << ")" << std::endl;
    std::cout << "Value of f at minimum: " << f(minimum) << std::endl;

    return 0;
}
