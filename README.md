# Intelligent-Placer

## План
1. Поиск границ листа и фигуры [Сделано]
2. Поиск границ предметов [Алгоритм в разработке]
3. Ответ на вопрос из постановки задачи [В разработке]

## Постановка задачи

### Описание задачи

По поданной на вход фотографии нескольких предметов и многоугольнику, которые расположены на белых листах бумаги, лежащих на ярко-зеленой поверхности, понимать, можно ли расположить одновременно все эти предметы на плоскости так, чтобы они влезли в этот многоугольник. Предметы и горизонтальная поверхность, которые могут оказаться на фотографии, заранее известны. 

### Входные и выходные значения

##### На вход:

Изображение в формате jpeg предеметов на белом листе бумаги и многоугольника, который нарисован темным маркером на белом листе бумаги. На фотографии оба листа бумаги лежат на ярко-зеленой поверхности. Лист бумаги с многоугольником всегда находится ниже листа бумаги, на котором лежат предметы.

##### На выход:

Выдаёт **True** - если данный набор предметов может поместиться на данный многоугольник и **False** - в ином случае. Ответ, в зависимости от конфигурации, может быть получен, как в консоль, так и в файл.



### Требования

##### К фотографиями:
- Сняты на одном устройстве
- На широкоугольную камеру 12 мп
- При одинаковом освещении
- Цветные, без наложения каких-то фильтров
- Фокус должен быть на предмете
- Без размытостей фона
- Отсутсвуют пересвеченные и серо-черные области
- Устройство находится под прямым углом к нормали поверхности
- Высота съемки - до 50 см

##### К предметам:
- Полностью помещаются на лист бумаги А4
- Полностью расположен на фото
- Не размываются с фоном(толщина линии границы не более 10 px)
- Не накладываются на другие предметы
- Нет одинаковых предметов

##### К поверхности:
- Светлая
- Горизонтальная
- Прямая
- Однотонная, без рисунков
- Отличается от цвета листа А4, на котором лежат предметы(не белый)

### Данные

Все собранные данные можно посмотреть [здесь](https://drive.google.com/drive/folders/1lh-NmPbrZmleoe6-WLharnoG1Y7TRa0L?usp=sharing)
