import torch

### 2.1 Простые вычисления с градиентами (8 баллов)
# Создайте тензоры x, y, z с requires_grad=True
# Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
# Найдите градиенты по всем переменным
# Проверьте результат аналитически
def task_2_1():
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    z = torch.tensor(4.0, requires_grad=True)
    
    f = x**2 + y**2 + z**2 + 2*x*y*z
    f.backward()
    
    x_grad_analytic, y_grad_analytic, z_grad_analytic = 28, 22, 20 # найдено аналитически
    if x.grad.item() == x_grad_analytic:
        print(f"Градиент x совпал, {x.grad}")
        
    if y.grad.item() == y_grad_analytic:
        print(f"Градиент y совпал, {y.grad}")
        
    if z.grad.item() == z_grad_analytic:
        print(f"Градиент z совпал, {z.grad}")
    
### 2.2 Градиент функции потерь (9 баллов)
# Реализуйте функцию MSE (Mean Squared Error):
# MSE = (1/n) * Σ(y_pred - y_true)^2
# где y_pred = w * x + b (линейная функция)
# Найдите градиенты по w и b
def task_2_2():
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([2, 4, 6])
    w = torch.tensor(5.0, requires_grad=True)
    b = torch.tensor(-6.0, requires_grad=True)
    
    y_pred = w * x + b
    
    mse = ((y_pred - y) ** 2).mean()
    mse.backward()
    
    print(f"MSE: {mse}")
    print(f"grad MSE по w: {w.grad}")
    print(f"grad MSE по b {b.grad}")

### 2.3 Цепное правило (8 баллов)
# Реализуйте составную функцию: f(x) = sin(x^2 + 1)
# Найдите градиент df/dx
# Проверьте результат с помощью torch.autograd.grad
def task_2_3():
    x = torch.tensor(2.0, requires_grad=True)
    
    f = torch.sin(x**2 + 1)
    grad_check = torch.autograd.grad(f, x)

    print(f"Проверка по torch.autograd.grad: {grad_check[0]}")

task_2_1()
task_2_2()
task_2_3()