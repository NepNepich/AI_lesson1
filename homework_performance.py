import torch
import time
### 3.1 Подготовка данных (5 баллов)
# Создайте большие матрицы размеров:
# - 64 x 1024 x 1024
# - 128 x 512 x 512
# - 256 x 256 x 256
# Заполните их случайными числами
A = torch.randn(64, 1024, 1024)
B = torch.randn(128, 512, 512)
C = torch.randn(256, 256, 256)
    
### 3.2 Функция измерения времени (5 баллов)
# Создайте функцию для измерения времени выполнения операций
# Используйте torch.cuda.Event() для точного измерения на GPU
# Используйте time.time() для измерения на CPU
def measure_time_gpu(func, repeats, *args):
    """Измерение времени выполнения функции на GPU"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # прогрев, как говорилось в лекции
    for _ in range(3):
        func(*args)
    
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeats):
        func(*args)
        
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / repeats # в миллисекундах

def measure_time_cpu(func, repeats, *args):
    """Измерение времени выполнения функции на CPU"""
    # прогрев, как говорилось в лекции
    for _ in range(3):
        func(*args)
    
    start_time = time.time()
    for _ in range(repeats):
        func(*args)
        
    end_time = time.time()
    
    return (end_time - start_time) / repeats * 1000 # в миллисекундах

### 3.3 Сравнение операций (10 баллов)
# Сравните время выполнения следующих операций на CPU и CUDA:
# - Матричное умножение (torch.matmul)
# - Поэлементное сложение
# - Поэлементное умножение
# - Транспонирование
# - Вычисление суммы всех элементов
# Для каждой операции:
# 1. Измерьте время на CPU
# 2. Измерьте время на GPU (если доступен)
# 3. Вычислите ускорение (speedup)
# 4. Выведите результаты в табличном виде
### Пример вывода:
# Операция          | CPU (мс) | GPU (мс) | Ускорение
# Матричное умножение|   150.2  |    12.3  |   12.2x
# Сложение          |    45.1  |     3.2  |   14.1x

def get_device():
    """Проверка доступности CUDA"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    print("CUDA не найдены")
    
    return torch.device("cpu")

def run_benchmark(*tensors : torch.Tensor, repeats):
    """3.1 - 3.3 Подготовка данных и сравнение операций."""
    device = get_device()
    operations = {
        "Матричное умножение": lambda A, B: torch.matmul(A, B),
        "Поэлементное сложение": lambda A, B: A + B,
        "Поэлементное умножение": lambda A, B: A * B,
        "Транспонирование": lambda A, B: A.transpose(1, 2),
        "Сумма элементов": lambda A, B: torch.sum(A)
    }
    
    print(f"{"Операция":<22} | {"Размер":<15} | {"CPU (мс)":<10} | {"GPU (мс)":<10} | {"Ускорение":<10}")
    for tensor in tensors:
        shape_str = f"{tensor.size(0)}x{tensor.size(1)}x{tensor.size(2)}"
        tensor_copy = tensor
        tensor_gpu = tensor.to(device) if device.type == "cuda" else None
        tensor_copy_gpu = tensor_copy.to(device) if device.type == "cuda" else None
        for op_name, op_func in operations.items():
            cpu_time = measure_time_cpu(op_func, repeats, tensor, tensor_copy)
            if device.type == "cuda":
                gpu_time = measure_time_gpu(op_func, repeats, tensor_gpu, tensor_copy_gpu)
                speedup = cpu_time / gpu_time if gpu_time > 0 else "много"
                gpu_time_str = f"{gpu_time}"
                speedup_str = f"{speedup}x"
            else:
                gpu_time_str = "нет"
                speedup_str = "нет"
                
            print(f"{op_name:<22} | {shape_str:<15} | {cpu_time:<10.2f} | {gpu_time_str:<10} | {speedup_str:<10}")
            
run_benchmark(A, B, C, repeats=5)

### 3.4 Анализ результатов (5 баллов)
# Проанализируйте результаты:
# - Какие операции получают наибольшее ускорение на GPU?
# - Почему некоторые операции могут быть медленнее на GPU?
# - Как размер матриц влияет на ускорение?
# - Что происходит при передаче данных между CPU и GPU?

# 1) Проанализировать скорость на GPU не могу, так как у меня видеокарта AMD
# Однако в интернете пишут, что матричное умножение и поэлементные операции обычно получают наибольшее ускорение на GPU,
# особенно при больших размерах матриц
# 2) ГПУ расчитаны для большого числа вычислений, но в случае, например, транспонирования, где вычисления особо нет
# главным ограничением становится работа с памятью. Ну а с памятью лучше работает цп. Похожая ситуация была бы,
# если бы у нас было много маленьких матриц
# 3) Скорость выполнения от 64x1024x1024 до 256x256x256, по крайней мере, на цп уменьшается.
# Размер матриц меньше, их число больше, вполне очевидно
# 4) Передача данных идет по pci шине, которая вполне может стать ботлнеком из-за пропускной способности