# Обработчик экспериментальных данных по кинетике затухания люминофоров

## О чём это

Создан для упрощения обработки экспериментальных данных в домашнем практикуме по физхимии для студентов 4 курса ХФ МГУ
(см методичку - `Кинетика_затухания_люминесценции_кристаллофосфоров.pdf`)
Но, в целом, может быть применён к любому видео, для которого надо получить зависимость яркости некоторого объекта от времени

## Установка

### Из исходного кода через архив - windows
Скачайте и разархивируйте

<https://drive.google.com/file/d/1tDp3CFueX802RgH7nDrI4nC-e6fJirer/view?usp=sharing>

### Из исходного кода через архив - linux

Пока нет

### Исполняемый файл - windows

Пока нет

### Исполняемый файл - linux

Пока нет

### Из исходного кода в лоб (100% рабочий, но запарный вариант)

1. Поставьте python 3.10-3.14 и poetry
Например, через conda (не забудьте добавить conda-poetry) или напрямую по ссылкам:
[python](https://www.python.org/downloads/)
[poetry](https://python-poetry.org/docs/#installation)
Проверьте, что всё добавлено в PATH
2. Скачайте проект и зависимости

```shell
git clone https://github.com/OstarkovSN/phos
cd phos
poetry install --no-root
```

## Запуск

Встаньте в рабочую папку проекта

На windows - Win+R, вбиваете cmd, открывается терминал, затем cd <путь к папке с исполняемым файлом/кодом>

На linux - сами знаете

Если вы ставили архив:

Windows:

```shell
main <путь к видео> [ОПЦИИ](см ниже )
```

Если вы устанавливали исполняемый файл:

Windows:

```shell
main.exe <путь к видео> [ОПЦИИ](см ниже )
```

Linux:

```shell
main <путь к видео> [ОПЦИИ](см ниже )
```

При установке из исходного кода:

```shell
poetry run python main.py <путь к видео> [ОПЦИИ](см ниже )
```

## Опции

```plain
-o, --output_path TEXT     Папка, куда сохранять
-p, --output_postfix TEXT  Постфикс для папки куда сохранять, работает
                            если output_path не предоставлен. Всё будет сохранено в
                            ${video_directory}/${output_postfix}
-c, --config_path TEXT     Путь к конфигу (см. ниже)
-v, --verbose              Больше информационных сообщений
-q, --quiet                Меньше информационных сообщений
--help                     Выводит аналогичное описание опций
```

## Конфиг

По умолчаниюю - config.yaml в папке с исп.файлом/исх кодом. Если его нет, он сгенерится при первом прогоне
Можно редактировать, чтобы получить другую окраску графиков. Лень прописывать ключи, разберётесь на месте.
HINT: внутри всего - plotly, читайте их документацию по go.Scatter

## Автор

Остарков С.Н., <ostarkovstepan@gmail.com>
Принимаются донаты в форме сидра или медовухи)
