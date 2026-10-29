import torch
### 1.1 Создание тензоров (7 баллов)
# Создайте следующие тензоры:
# - Тензор размером 3x4, заполненный случайными числами от 0 до 1
# - Тензор размером 2x3x4, заполненный нулями
# - Тензор размером 5x5, заполненный единицами
# - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
def task_1_1():
    tensor_rand = torch.rand(3, 4)
    tensor_zeros = torch.zeros(2, 3, 4)
    tensor_ones = torch.ones(5, 5)
    tensor_range = torch.arange(16).reshape(4, 4)
    
    return tensor_rand, tensor_zeros, tensor_ones, tensor_range

### 1.2 Операции с тензорами (6 баллов)
# Дано: тензор A размером 3x4 и тензор B размером 4x3
# Выполните:
# - Транспонирование тензора A
# - Матричное умножение A и B
# - Поэлементное умножение A и транспонированного B
# - Вычислите сумму всех элементов тензора A
def task_1_2():
    A = torch.rand(3, 4)
    B = torch.rand(4, 3)

    A_T = A.transpose(0, 1)
    AB_matmul = torch.matmul(A, B)
    AB_T_elem_mul = A * B.transpose(0, 1)
    A_sum = torch.sum(A)

    return A_T, AB_matmul, AB_T_elem_mul, A_sum

### 1.3 Индексация и срезы (6 баллов)
# Создайте тензор размером 5x5x5
# Извлеките:
# - Первую строку
# - Последний столбец
# - Подматрицу размером 2x2 из центра тензора
# - Все элементы с четными индексами
def task_1_3():
    A = torch.arange(125).reshape(5, 5, 5)
    
    first_row = A[0, 0, :]
    last_column = A[-1, :, -1]
    center_2x2 = A[2, 1:3, 1:3] 
    even_idx_values = A[:, :, ::2] # если правильно понял, о каких четных индексах речь
    
    return first_row, last_column, center_2x2, even_idx_values


### 1.4 Работа с формами (6 баллов)
# Создайте тензор размером 24 элемента
# Преобразуйте его в формы:
# - 2x12
# - 3x8
# - 4x6
# - 2x3x4
# - 2x2x2x3
def task_1_4():
    A = torch.arange(24)
    
    A_2x12 = A.reshape(2, 12)
    A_3x8 = A.reshape(3, 8)
    A_4x6 = A.reshape(4, 6)
    A_2x3x4 = A.reshape(2, 3, 4)
    A_2x2x2x3 = A.reshape(2, 2, 2, 3)
    
    return A_2x12, A_3x8, A_4x6, A_2x3x4, A_2x2x2x3

if __name__ == "__main__":
    print(task_1_3())