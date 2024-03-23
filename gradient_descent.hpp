#ifndef GRADIENT_DESCENT_HPP
#define GRADIENT_DESCENT_HPP

#include <Eigen/Dense>
#include <functional>

// PARAMETERS --------------------------------------------------------
struct OptParams {
    Eigen::VectorXd initial_guess;
    double epsilon_r;
    double epsilon_s;
    int max_iterations;
};

struct DecayParams {
    double alpha0;
    double mu = 0;
    double sigma = 0;
};

// LEARNING DECAY STRATEGY -------------------------------------------
class LearningRateDecay {
protected:
    DecayParams dec_params;
public:
    LearningRateDecay() = default;
    LearningRateDecay(const DecayParams& dec_params) : dec_params(dec_params) {}
    virtual ~LearningRateDecay() = default;
    virtual double computeStepSize(int iteration) = 0;
    virtual std::string getName() const = 0;

    DecayParams getDecayParams() const {
        return dec_params;
    }

    virtual void set_x(const Eigen::VectorXd& x) {
        // Do nothing
    }
};

// Constant Learning Rate
class NoDecay : public LearningRateDecay {
public:
    NoDecay() = default;
    NoDecay(const DecayParams& dec_params) : LearningRateDecay(dec_params) {}
    double computeStepSize(int iteration) override {
        return dec_params.alpha0;
    }

    std::string getName() const {
        return "NoDecay";
    }
};

// Exponential Learning Rate Decay
class ExponentialDecay : public LearningRateDecay {
public:
    ExponentialDecay() = default;
    ExponentialDecay(const DecayParams& dec_params) : LearningRateDecay(dec_params) {}
    double computeStepSize(int iteration) override {
        return dec_params.alpha0 * std::exp(-dec_params.mu * iteration);
    }

    std::string getName() const {
        return "ExponentialDecay";
    }
};

// Inverse Learning Rate Decay
class InverseDecay : public LearningRateDecay {
public:
    InverseDecay() = default;
    InverseDecay(const DecayParams& dec_params) : LearningRateDecay(dec_params) {}
    double computeStepSize(int iteration) override {
        return dec_params.alpha0 / (1 + dec_params.mu * iteration);
    }

    std::string getName() const {
        return "InverseDecay";
    }
};

// Armijo Rule for Learning Rate Decay
class ArmijoDecay : public LearningRateDecay {
    std::function<double(const Eigen::VectorXd&)> f;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> grad_f;
    Eigen::VectorXd x;

public:
    ArmijoDecay() = default;
    ArmijoDecay(const DecayParams& dec_params,
               const std::function<double(const Eigen::VectorXd&)>& f,
               const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& grad_f) 
               : LearningRateDecay(dec_params), f(f), grad_f(grad_f) {}

    double computeStepSize(int iteration) override {
        double f_x = f(x);
        Eigen::VectorXd grad = grad_f(x);
        double norm = grad.norm();

        double decrease_threshold = dec_params.sigma * dec_params.alpha0 * norm*norm;

        double alpha_temp = dec_params.alpha0;
        
        // Sufficent decrease condition
        while (f_x - f(x - alpha_temp * grad) < decrease_threshold) {
            alpha_temp *= 0.5;
            decrease_threshold = dec_params.sigma * alpha_temp * norm*norm;
            }
        return alpha_temp;
    }

    void set_x(const Eigen::VectorXd& x) override {
        this->x = x;
    }

    std::string getName() const {
        return "ArmijoDecay";
    }
};

// OPTIMIZATION METHODS ----------------------------------------------
class Optimizer {
protected:
    int iteration = 0;
    std::function<double(const Eigen::VectorXd&)> f;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> grad_f;
    OptParams opt_params;
    Eigen::VectorXd x;

    std::shared_ptr<LearningRateDecay> lr_decay = nullptr;
    
    bool checkConvergence(const Eigen::VectorXd& new_x, const Eigen::VectorXd& x,
                          const double epsilon_r, const double epsilon_s,
                          const double f_new, const double f_x) {
        return ((new_x - x).norm() < epsilon_s || std::abs(f_new - f_x) < epsilon_r);
    }

    void incrementIteration() { ++iteration; }

public:
    Optimizer() = default;
    Optimizer(const std::function<double(const Eigen::VectorXd&)>& f,
              const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& grad_f,
              const OptParams& opt_params,
              std::shared_ptr<LearningRateDecay> lr_decay) 
              : f(f), grad_f(grad_f), opt_params(opt_params) {
                    x = opt_params.initial_guess;
                    this->lr_decay = std::move(lr_decay);
              }

    virtual ~Optimizer() = default;
    
    virtual Eigen::VectorXd minimize() = 0;
};

// Standard Gradient Descent
class GradientDescent : public Optimizer {
public:
    GradientDescent() = default;
    GradientDescent(const std::function<double(const Eigen::VectorXd&)>& f,
                    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& grad_f,
                    const OptParams& opt_params,
                    std::shared_ptr<LearningRateDecay> lr_decay) 
                    : Optimizer(f, grad_f, opt_params, std::move(lr_decay)) {}
    
    Eigen::VectorXd minimize() override;
};

// Momentum (or Heavy Ball) Method
class Momentum : public Optimizer {
protected:
    void checkLearningRateDecay() {
        if (lr_decay->getName() == "ArmijoDecay") {
            std::cout << "Invalid strategy (Armijo Rule) for Momentum methods.\n" 
                      << "Setting the default constant learning rate.\n" << std::endl;
            lr_decay = std::make_unique<NoDecay>(lr_decay->getDecayParams());
        }
    }

    double eta = 0.9; // Momentum parameter
public:
    Momentum() = default;
    Momentum(const std::function<double(const Eigen::VectorXd&)>& f,
             const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& grad_f,
             const OptParams& opt_params,
             std::shared_ptr<LearningRateDecay> lr_decay) 
             : Optimizer(f, grad_f, opt_params, std::move(lr_decay)) {
                 checkLearningRateDecay();
             }
    virtual Eigen::VectorXd minimize() override;
};

// Nesterov Method
class Nesterov : public Momentum {
public:
    Nesterov() = default;
    Nesterov(const std::function<double(const Eigen::VectorXd&)>& f,
             const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& grad_f,
             const OptParams& opt_params,
             std::shared_ptr<LearningRateDecay> lr_decay) 
             : Momentum(f, grad_f, opt_params, std::move(lr_decay)) {}
    
    Eigen::VectorXd minimize() override;
};

#endif // GRADIENT_DESCENT_HPP