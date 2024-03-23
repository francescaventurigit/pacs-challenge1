#include <Eigen/Dense>
#include <functional>
#include <iostream>
#include <memory>
#include "gradient_descent.hpp"

Eigen::VectorXd GradientDescent::minimize() {

    // First iteration
    Eigen::VectorXd new_x = x - lr_decay->getDecayParams().alpha0 * grad_f(x);
    x = new_x;
    incrementIteration();

    // Following iterations
    while (iteration < opt_params.max_iterations) {
        
        // Compute step size according to the learning rate decay strategy
        if (lr_decay->getName() == "ArmijoDecay") {
            lr_decay->set_x(x);
            }
        double alpha = lr_decay->computeStepSize(iteration);

        // Update x
        new_x = x - alpha * grad_f(x);
        
        // Check convergence
        if (checkConvergence(new_x, x, opt_params.epsilon_r, opt_params.epsilon_s, f(new_x), f(x))) {
            std::cout << "Convergence achieved at iteration " << iteration << std::endl;
            return new_x;
        }
        x = new_x;

        incrementIteration();
    }
    std::cout << "Maximum number of iteration reached: " << iteration << std::endl;
    return new_x;
}

Eigen::VectorXd Momentum::minimize() {

    // First iteration
    Eigen::VectorXd prev_x = x;
    Eigen::VectorXd new_x = x - lr_decay->getDecayParams().alpha0 * grad_f(x);
    x = new_x;
    incrementIteration();

    // Following iterations
    while (iteration < opt_params.max_iterations) {
        
        // Compute step size according to the learning rate decay strategy
        double alpha = lr_decay->computeStepSize(iteration);
        
        // Update x
        new_x = x - alpha * grad_f(x) + eta * (x - prev_x);
        
        // Check convergence
        if (checkConvergence(new_x, x, opt_params.epsilon_r, opt_params.epsilon_s, f(new_x), f(x))) {
            std::cout << "Convergence achieved at iteration " << iteration << std::endl;
            return new_x;
        }

        prev_x = x;
        x = new_x;

        incrementIteration();
    }
    std::cout << "Maximum number of iteration reached: " << iteration << std::endl;
    return x;
}

Eigen::VectorXd Nesterov::minimize() {

    // First iteration
    Eigen::VectorXd prev_x = x;
    Eigen::VectorXd new_x = x - lr_decay->getDecayParams().alpha0 * grad_f(x);
    incrementIteration();

    while (iteration < opt_params.max_iterations) {
        // Compute step size according to the learning rate decay strategy
        double alpha = lr_decay->computeStepSize(iteration);

        // Update x
        Eigen::VectorXd y = x + eta * (x - prev_x);
        new_x = y - alpha * grad_f(y);

        // Check convergence
        if (checkConvergence(new_x, x, opt_params.epsilon_r, opt_params.epsilon_s, f(new_x), f(x))) {
            std::cout << "Convergence achieved at iteration: " << iteration << std::endl;
            return new_x;
        }
    
        prev_x = x;
        x = new_x;

        incrementIteration();
    }
    std::cout << "Maximum number of iteration reached: " << iteration << std::endl;

    return x;
}