# Лабораторная работa No1
# Реализация метода обратного распространения ошибки для двуслойной полностью связанной нейронной сети

### Вывод математических формул

![dnn-diagram](dnn-diagram.png)

###### [вывод формул](./deriving-formulas.png)

А каждом слое есть веса данные матрицой
![w  \in \mathbb{R}^{(n\_outputs,n_inputs)}](https://latex.codecogs.com/svg.latex?w&space;\in&space;\mathbb{R}^{(n\_outputs,n_inputs)})
где s обазначивает слой.

На изображении не показаны узлы и веса смешения, потому что всё проще если добавлять единицу в входные векторы. В конце из всех формул легко выделяем части связанные с смещением, а это понадобится для программной реализации.

![u_k = \sum_{j=1}^J{w_{k,j}x_j}, v_k = u_k + b_k = \sum_{j=0}^J{w_{k,j}x_j}](https://latex.codecogs.com/svg.latex?u_k&space;=&space;\sum_{j=1}^J{w_{k,j}x_j},&space;v_k&space;=&space;u_k&space;&plus;&space;b_k&space;=&space;\sum_{j=0}^J{w_{k,j}x_j})

![a_k = \phi(u_k + b_k) = \phi(v_k)](https://latex.codecogs.com/svg.latex?a_k&space;=&space;\phi(u_k&space;&plus;&space;b_k)&space;=&space;\phi(v_k))

Е представляет функцию ошибки, целевую функцию, которую хотим минимизировать.

![E(\phi^{(h)}(w^{(h)}a^{(h-1)}))) = E(\phi^{(h)}(w^{(h)}\phi^{(h-1)}(w^{(h-1)}a^{(h-2)})))](https://latex.codecogs.com/svg.latex?E(\phi^{(h)}(w^{(h)}a^{(h-1)}))=E(\phi^{(h)}(w^{(h)}\phi^{(h-1)}(w^{(h-1)}a^{(h-2)}))))

![a^{(0)}=x](https://latex.codecogs.com/svg.latex?a^{(0)}=x)

![\frac{\partial E}{\partial w^{(h)}_{i,j}} = \frac{\partial E}{\partial a^{(h)}_{i}}\frac{\partial a^{(h)}_{i}}{\partial v^{(h)}_{i}}\frac{\partial v^{(h)}_{i}}{\partial w^{(h)}_{i,j}} = \frac{\partial E}{\partial v^{(h)}_{i}}a^{(h-1)}_j = \delta^{(h)}a^{(h-1)}_j](https://latex.codecogs.com/svg.latex?\frac{\partial&space;E}{\partial&space;w^{(h)}_{i,j}}&space;=&space;\frac{\partial&space;E}{\partial&space;a^{(h)}_{i}}\frac{\partial&space;a^{(h)}_{i}}{\partial&space;v^{(h)}_{i}}\frac{\partial&space;v^{(h)}_{i}}{\partial&space;w^{(h)}_{i,j}}&space;=&space;\frac{\partial&space;E}{\partial&space;v^{(h)}_{i}}a^{(h-1)}_j&space;=&space;\delta^{(h)}a^{(h-1)}_j)


![\frac{\partial E}{\partial w^{(h)}_{i,j}} = \sum_k\frac{\partial E}{\partial a^{(h)}_{k}}\frac{\partial a^{(h)}_{k}}{\partial v^{(h)}_{k}}\frac{\partial v^{(h)}_{k}}{\partial a^{(h-1)}_i}\frac{\partial a^{(h-1)}_i}{\partial v^{(h-1)}_i}\frac{\partial v^{(h-1)}_i}{\partial w^{(h)}_{i,j}} = \frac{\partial E}{\partial v^{(h-1)}_i}a^{(h-1)}_j = \delta^{(h-1)}_i a^{(h-2)}_j](https://latex.codecogs.com/svg.latex?\frac{\partial&space;E}{\partial&space;w^{(h)}_{i,j}}&space;=&space;\sum_k\frac{\partial&space;E}{\partial&space;a^{(h)}_{k}}\frac{\partial&space;a^{(h)}_{k}}{\partial&space;v^{(h)}_{k}}\frac{\partial&space;v^{(h)}_{k}}{\partial&space;a^{(h-1)}_i}\frac{\partial&space;a^{(h-1)}_i}{\partial&space;v^{(h-1)}_i}\frac{\partial&space;v^{(h-1)}_i}{\partial&space;w^{(h)}_{i,j}}&space;=&space;\frac{\partial&space;E}{\partial&space;v^{(h-1)}_i}a^{(h-1)}_j&space;=&space;\delta^{(h-1)}_i&space;a^{(h-2)}_j)

Продолжая выводить формулы для последовательных слоев,  получаем общую формулу.

![\delta^{(s-1)}_i = \frac{\partial E}{\partial v^{(s-1)}_i} = \phi^{(s-1)'}(v^{(s-1)}_i)\sum_k\delta^{(s)}_kw^{(s)}_{k,i}
](https://latex.codecogs.com/svg.latex?\delta^{(s-1)}_i&space;=&space;\frac{\partial&space;E}{\partial&space;v^{(s-1)}_i}&space;=&space;\phi^{(s-1)'}(v^{(s-1)}_i)\sum_k\delta^{(s)}_kw^{(s)}_{k,i})

![ \delta^{(h)}_i = \frac{\partial E}{\partial v^{(h)}_i}](https://latex.codecogs.com/svg.latex?\delta^{(h)}_i&space;=&space;\frac{\partial&space;E}{\partial&space;v^{(h)}_i})

![\frac{\partial E}{\partial w^{(s)}_{k,i}} = \delta^{(s)}_i a^{(s-1)}_j](https://latex.codecogs.com/svg.latex?\frac{\partial&space;E}{\partial&space;w^{(s)}_{k,i}}&space;=&space;\delta^{(s)}_i&space;a^{(s-1)}_j)

Если использовать softmax функцию активации и кросс-энтропийную функцию ошибки, получаем следующие формулы:
Softmax активация:
![\phi^{(h)}(v^{(h)}_i) = \frac{e^{v^{(h)}_i}}{\sum_ke^{v^{(h)}_k}}](https://latex.codecogs.com/svg.latex?\phi^{(h)}(v^{(h)}_i)&space;=&space;\frac{e^{v^{(h)}_i}}{\sum_ke^{v^{(h)}_k}})

Softmax это апостериорная вероятность того, что пример тренировочной
выборки принадлежит данному классу.

![p(Y|W) = \prod^N_{n=1}\prod^K_{j=1}(\frac{e^{v^{(h)}_i}}{\sum_ke^{v^{(h)}_k}})^{y^{(n)}_j}](https://latex.codecogs.com/svg.latex?p(Y|W)&space;=&space;\prod^N_{n=1}\prod^K_{j=1}(\frac{e^{v^{(h)}_i}}{\sum_ke^{v^{(h)}_k}})^{y^{(n)}_j})

Максимизация функции правдоподобия может быть сведена к минимизации логарифма.

![E(W) = -1/N\ln\prod^N_{n=1}\prod^K_{j=1}(\frac{e^{v^{(h)}_i}}{\sum_ke^{v^{(h)}_k}})^{y^{(n)}_j} = -1/N\sum^N_{n=1}\sum^K_{j=1}y^{(n)}_j\frac{e^{v^{(h)}_i}}{\sum_ke^{v^{(h)}_k}}](https://latex.codecogs.com/svg.latex?E(W)&space;=&space;-1/N\ln\prod^N_{n=1}\prod^K_{j=1}(\frac{e^{v^{(h)}_i}}{\sum_ke^{v^{(h)}_k}})^{y^{(n)}_j}&space;=&space;-1/N\sum^N_{n=1}\sum^K_{j=1}y^{(n)}_j\frac{e^{v^{(h)}_i}}{\sum_ke^{v^{(h)}_k}})

![\frac{\partial E}{\partial v^{(h)}_i} = \sum^K_{j=1, j \neq i}(y_j\frac{e^{v^{(h)}_j}}{\sum_ke^{v^{(h)}_k}}) + y_i\frac{e^{v^{(h)}_i}}{\sum_ke^{v^{(h)}_k}} - y_i = \sum^K_{j=1}(y_j\frac{e^{v^{(h)}_j}}{\sum_ke^{v^{(h)}_k}}) - y_i = \frac{e^{v^{(h)}_j}}{\sum_ke^{v^{(h)}_k}}) - y_i = a^{(h)}_i - y_i = \delta^{(h)}_i](https://latex.codecogs.com/svg.latex?\frac{\partial&space;E}{\partial&space;u^{(h)}_i}&space;=&space;\sum^K_{j=1,&space;j&space;\neq&space;i}(y_j\frac{e^{v^{(h)}_j}}{\sum_ke^{v^{(h)}_k}})&space;&plus;&space;y_i\frac{e^{v^{(h)}_i}}{\sum_ke^{v^{(h)}_k}}&space;-&space;y_i&space;=&space;\sum^K_{j=1}(y_j\frac{e^{v^{(h)}_j}}{\sum_ke^{v^{(h)}_k}}\)&space;-&space;y_i&space;=&space;\frac{e^{v^{(h)}_j}}{\sum_ke^{v^{(h)}_k}})&space;-&space;y_i&space;=&space;a^{(h)}_i&space;-&space;y_i&space;=&space;\delta^{(h)}_i)

Теперь, чтобы запрограммировать обратное распространение ошибки, нужно ещё только производную функции ReLU `(ReLU(x) = max(x, 0))`, которую мы будем использовать.

`ReLU'(x) = I(x > 0) = 1{x > 0}`

### Псевдокод алгоритма

**(Скорость обучения и полученный результат много зависят от инициализации весов.)**

```
Инициализировать веса w.

Пока не выполнены условия остановки:
  Для каждой пачки тренировочных примеров:
    Для каждого тренировочного примера:
      Вычисляем функцию ошибки по формуле
```
.![E(\phi^{(h)}(w^{(h)}\phi^{(h-1)}(w^{(h-1)}a^{(h-2)})))=E(a^{(h)})=-1/N_{batch}\sum^N_{batch}_{n=1}\sum^K_{j=1}y^{(n)}_ja^{(h,n)}](https://latex.codecogs.com/svg.latex?E(\phi^{(h)}(w^{(h)}\phi^{(h-1)}(w^{(h-1)}a^{(h-2)})))=E(a^{(h)})=-1/N_{batch}\sum^{N_{batch}}_{n=1}\sum^K_{j=1}y^{(n)}_ja^{(h,n)})
```
      Промежуточные результаты используем чтобы вычислить градиенты.
```
![\delta^{(h)}_i=\frac{\partial E}{\partial v^{(h)}_i}=a^{(h)}_i-y_i](https://latex.codecogs.com/svg.latex?\delta^{(h)}_i=\frac{\partial&space;E}{\partial&space;v^{(h)}_i}=a^{(h)}_i-y_i)
![\delta^{(s-1)}_i = \frac{\partial E}{\partial v^{(s-1)}_i} = \phi^{(s-1)'}(v^{(s-1)}_i)\sum_k\delta^{(s)}_kw^{(s)}_{k,i}
](https://latex.codecogs.com/svg.latex?\delta^{(s-1)}_i&space;=&space;\frac{\partial&space;E}{\partial&space;v^{(s-1)}_i}&space;=&space;\phi^{(s-1)'}(v^{(s-1)}_i)\sum_k\delta^{(s)}_kw^{(s)}_{k,i})
![\frac{\partial E}{\partial w^{(s)}_{k,i}} = \delta^{(s)}_i a^{(s-1)}_j](https://latex.codecogs.com/svg.latex?\frac{\partial&space;E}{\partial&space;w^{(s)}_{k,i}}&space;=&space;\delta^{(s)}_i&space;a^{(s-1)}_j)
```
      Вычисляем градиент по всей пачке
```
![\Delta w^{(s)}_{i,j;batch}+=\delta^{(s)}_i a^{(s-1)}_j](https://latex.codecogs.com/svg.latex?\Delta&space;w^{(s)}_{i,j;batch}&plus;=\delta^{(s)}_i&space;a^{(s-1)}_j)
```
  Используя градиентный спуск обновляем веса
```
![w^{(s)}_{i,j} = w^{(s)}_{i,j} - \alpha \Delta w^{(s)}_{i,j;batch}](https://latex.codecogs.com/svg.latex?w^{(s)}_{i,j}&space;=&space;w^{(s)}_{i,j}&space;-&space;\alpha&space;\Delta&space;w^{(s)}_{i,j;batch})
```
Повтаряем
```

### Описание реализации

Код распределён на несколько файлов. Вспомогательные функции в utils.py и data_loader.py (вчитывание данных). Алгоритм нейронной сети запрограммирован в dnn.py. Он представлен классом Classifier, объекты которого можно настроить различными слоями (файл layers.py), функциями активации (activations.py) и функциями ошибки (cost.py). Слой SoftmaxCrossEntropyLayer должен использоваться в конце сети вместе с функцией ошибки softmax_cross_entropy.

### Как запустить

Сначала рекомендуется установить python3, и питоновские пакеты numpy и необязательно matplotlib. Для этого удобно запустить `configure.sh`, если пользуетесь линуксом на базе Debian (Debian, Ubuntu, Mint,...).

Kоманда `./main.py -h` показывает возможные настройки через командную строку:
```
usage: main.py [-h] [-d DATADIR] [-s SIZES] [-l LEARNING_RATE] [-m MAX_ITER]
               [-b BATCH_SIZE] [-c]

optional arguments:
  -h, --help            show this help message and exit
  -d DATADIR, --dir DATADIR
                        path to directory containing data
  -s SIZES, --hidden-layer-sizes SIZES
                        sizes of hidden layers separated by commas, e.g.
                        '128,256'
  -l LEARNING_RATE, --learning-rate LEARNING_RATE
                        learning rate
  -m MAX_ITER, --max-iter MAX_ITER
                        maximum number of iterations
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size
  -c, --save-learning-curve
                        dumps learning curve graph to learning_curve.png
```

Kоманда `./main.py -m 50 -s 96 -l 0.1 -b 200 -d ../datasets/mnist/ -c` обозначает что мв хотим натренировать сеть с одним скрытым слоем и 96 нейронов в этом слое, при этом используются максимально 50 итераций, скорость обучения 0.1 и пачки по 200 примеров, а данные находятся в папке ../datasets/mnist. Чтобы сохранить граф кривой обучения используются -c.

Такая команда даст нам следующие результаты/точности:

| на тренировочной выборке | на тестовой выборке |
| ------------------------:| -------------------:|
|          99.28%          |        97.58%       |

с такой кривой обучения:
![learning-curve](./learning_curve.png)
