import pandas as pd
import inspect
import numpy as np
import os
import datetime
import math
from cmd import Cmd
from dateutil.relativedelta import relativedelta
date=None
base_name = {
    'df_main_price': {
        'path':'base/База по ценам Декарт.xlsx',
        "list":['Объект', 'Тип квартир'],
        "sheet":"Цены"},
    'df_house_price': {
        'path':'base/База по ценам Дома.xlsx',
        "list":['id_house', 'Объект'],
        "sheet":"Цены"},
    'df_zk_price':{
        'path':'base/База по ценам ЖК.xlsx',
        "list":['id_zk', 'Объект'],
        "sheet":"Цены"} ,
    'df_main_sell': {
        'path':'base/База по продажам.xlsx',
        "list":['id_house', 'Объект','Тип квартир'],
        "sheet":"Продажи"},
    'df_zk_sell': {
        'path':'base/База по продажам ЖК.xlsx',
        "list":['id_zk', 'Объект'],
        "sheet":"Продажи"},
    'df_house_sell':{
        'path':'base/База по продажам Дома.xlsx',
        "list":['id_house', 'Объект'],
        "sheet":"Продажи"}
}
dict_id_zk = {}
database=None
df_sell = None
df_type = None
name = ""
# new_house_zk = False
for i in base_name:
    try:
        xl = pd.ExcelFile(base_name[i].get("path"))
        df1 = xl.parse(base_name[i].get("sheet"))
        if i=="df_main_price":
            df1 = df1.astype(object).replace({'—': np.nan, '-': np.nan})
            df1.iloc[:, 2:] = df1.iloc[:, 2:].apply(pd.to_numeric, errors='ignore')
        globals()[i] = df1.copy()
    except:
        globals()[i] = pd.DataFrame(columns=base_name[i].get("col"))

####################################################################
####################################################################
#Вспомогательные функции
def verb_to_str(x):
    ''' Вспомогательная функция для перевода имени переменной в строку'''
    caller_globals = inspect.currentframe().f_back.f_globals
    return [name for name, value in caller_globals.items() if x is value][0]

def csv_to_excel(name):
    '''Переводит из csv в excel'''
    if len(name.split('.')) == 1:
        os.mkdir(name + "_excel")
        dirs = os.listdir(name)
        lst_csv = sorted(dirs, key=lambda x: str(x.split(".")[0]))
        for file in lst_csv:
            df = pd.read_csv(name + "/" + file, sep=';', encoding='cp1251')
            writer = pd.ExcelWriter(name + "_excel/" + file.split(".")[0] + ".xlsx", engine='xlsxwriter')
            df.to_excel(writer, 'Цены')
            writer.save()
    else:
        writer = pd.ExcelWriter(name.split('.')[0].split('/')[-1] + ".xlsx", engine='xlsxwriter')
        df = pd.read_csv(name, sep=';', encoding='cp1251')
        df.to_excel(writer, 'Цены', index=False)
        writer.save()

def excel_to_csv(name):
    '''Парсинг из excel в csv'''
    if len(name.split('.')) == 1:
        os.mkdir(name + "_csv")
        dirs = os.listdir(name)
        lst_csv = sorted(dirs, key=lambda x: str(x.split(".")[0]))
        for file in lst_csv:
            xl = pd.ExcelFile(name + "/" + file)
            df1 = xl.parse('Цены')
            df1.to_csv(name + "_csv/" + file.split(".")[0] + '.csv', index=False, sep=';', encoding='cp1251')
    else:
        xl = pd.ExcelFile(name)
        df1 = xl.parse('Цены')
        df1.to_csv(name.split('.')[0].split('/')[-1] + '.csv', index=False, sep=';', encoding='cp1251')

def record_base_excel_quarter():
    '''Запись в соответствующие поквартальные базы в эксель'''
    for db in base_name:
        # print(globals()[db])
        df = average_quarter(globals()[db])
        writer = pd.ExcelWriter(base_name[db].get("path"))
        df.to_excel(writer, base_name[db].get("sheet"))
        writer.save()
def record_base_excel_month():
    '''Запись в соответствующие помесячные базы в  эксель'''
    for db in base_name:
        # print(globals()[db])
        df = average_columns(globals()[db])
        writer = pd.ExcelWriter(base_name[db].get("path"))
        df.to_excel(writer, base_name[db].get("sheet"), index=False)
        writer.save()

####################################################################
# def check_columns(df):
#     '''Проверка на присутствие или отсутсвие в df_main_price'''
#     global df_main_price
#     lst_index = list(df_main_price)
#     if df_main_price.empty:
#         df_main_price = df.copy()
#
#     else:
#         if isinstance(lst_index[-1], datetime.datetime) and (list(df)[-1].month>=lst_index[-1].month):
#             if ((lst_index[-1].month - list(df)[-1].month) == -1) and ((lst_index[-1].year - list(df)[-1].year) == 0):
#                 result = pd.merge(df_main_price, df, how='outer', on=base_name["df_main_price"].get("list"))
#                 df_main_price = result.copy()
#             elif ((lst_index[-1].month - list(df)[-1].month) == 0) and ((lst_index[-1].year - list(df)[-1].year) == 0):
#                 result = pd.merge(df_main_price, df, how='outer', on=base_name["df_main_price"].get("list"))
#                 df_main_price = result.copy()
#             else:
#                 df_main_price[lst_index[-1] + relativedelta(months=+1)] = np.nan
#                 return check_columns(df)
#         else:
#             result = pd.merge(df_main_price, df, how='outer', on=base_name["df_main_price"].get("list"))
#             df_main_price = result.copy()

def check_columns(df):
    '''Проверка на присутствие или отсутсвие в df_main_price'''
    global df_main_price
    lst_index = list(df_main_price)
    if df_main_price.empty:
        df_main_price = df.copy()
    elif list(df)[-1] in lst_index:
        return ""
    else:
        if isinstance(lst_index[-1], datetime.datetime) and (list(df)[-1].month>=lst_index[-1].month):
            if ((lst_index[-1].month - list(df)[-1].month) == -1) and ((lst_index[-1].year - list(df)[-1].year) == 0):
                result = pd.merge(df_main_price, df, how='outer', on=base_name["df_main_price"].get("list"))
                df_main_price = result.copy()
                # average_columns()
                return df
            elif ((lst_index[-1].month - list(df)[-1].month) == 0) and ((lst_index[-1].year - list(df)[-1].year) == 0):
                result = pd.merge(df_main_price, df, how='outer', on=base_name["df_main_price"].get("list"))
                df_main_price = result.copy()
            else:
                df_main_price[lst_index[-1] + relativedelta(months=+1)] = np.nan
                return check_columns(df)
        else:
            result = pd.merge(df_main_price, df, how='outer', on=base_name["df_main_price"].get("list"))
            df_main_price = result.copy()

def add_zk_house(df_main,df,col):
    id = base_name[verb_to_str(df_main)].get("list")[0]
    cols = base_name[verb_to_str(df_main)].get("list")
    df = df.rename(columns={
        # 'Название': 'Объект',
        col: date,
    })
    df_col = df[id.split(".") + ['Объект', date]]
    df_col = df_col.astype(object).replace({'—': np.nan, '-': np.nan})
    df_col.iloc[:, 2] = pd.to_numeric(df_col.iloc[:, 2])
    df_col[id] = df_col[id].astype(int)
    if df_main.empty:
        df_main = df_col.sort_values([id]).copy()
    elif not df_main.empty:
        result = pd.merge(df_main, df_col, how='outer', on=cols)
        df_main = result.copy()
    return df_main

def add_to_df_main_price(df):
    '''Форматирование таблицы с ключевым столбцом ЦЕНА
    для дальнейшего добавления к основной df_main_price'''
    # df_file = df.copy()
    global df_zk_price,df_house_price
    df = df.rename(columns={
        # 'Название': 'Объект',
        'Цена за кв.м., руб./кв.м.': date,
    })
    df1 = df[['Объект', 'Тип квартир', date]]
    df1.iloc[:, 2] = pd.to_numeric(df1.iloc[:, 2])
    # df_zk_price = add_zk_house(df_zk_price,df)
    # df_house_price = add_zk_house(df_house_price,df)
    df1 = df1.groupby(['Объект', 'Тип квартир'], sort=False,as_index=False).mean()
    check_columns(df1)

def is_nan(df):
    for index, row in df.iterrows():
        if not math.isnan(row[-1]):
            count = list(row.isnull()).index(False)
            while count < len(row):
                if math.isnan(row[count]):
                    row[count] = row[count - 1] + (row[-1] - row[count - 1]) / (len(row) - count + 1)
                count += 1
####################################################################
def fill_kvar_etap(df):
#     global df_type
#     df_type = df.groupby("Тип квартир").agg({
#         "Площадь, кв.м.": "mean",
# }).reset_index()
#     print(df_type)
    items = df_sell.to_dict('records')
    for index,rows in df.iterrows():
        for item in items:
            if rows["id_house"]==item["id_house"] and (item["Этап"]=="Сдан. Продаж Нет" or item["Этап"]=="Строится. Продаж Нет"):
                try:
                    df.loc[index, "Количество проданных квартир, шт."] = rows["Кол-во квартир по проектным декларациям, шт."]
                    df.loc[index,"Площадь проданных квартир, кв.м."] = \
                        rows["Кол-во квартир по проектным декларациям, шт."]*df_type.loc[df_type["Тип квартир"]==rows["Тип квартир"]]["Площадь, кв.м."].values[0]
                except:
                    pass
    return df

def average_quarter(df_main):
    df = df_main.set_index(['Навзвание', "Тип"])
    new = df.T.reset_index()  # транспонируем матрицу, чтобы столбцы-даты стали строками
    new['index'] = pd.PeriodIndex(new["index"], freq='Q')
    new = new.groupby(['index']).mean()  # групперуем по этому столбцу и считаем среднее
    df = new.T  # транспонируем обратно
    return df

def average_columns(df_main):
    df_name = verb_to_str(df_main)
    cols = base_name[df_name].get("list")
    data = df_main.loc[:, :cols[-1]].copy()
    df_main.drop(cols, axis="columns", inplace=True)
    df_main.columns = pd.Series(df_main.columns).apply(lambda x: x.to_period('M').to_timestamp())
    df_main = df_main.groupby(df_main.columns, axis=1).mean()
    # print(df_name)
    if df_name== "df_main_price":
        is_nan(df_main)
    df_main = data.join(df_main)
    return df_main

def create_bd_id(df):
    global database
    if "id_zk" not in df.columns.tolist():
        df = create_id_col_table(df,["id_zk", "Объект"])
    df1 = df[["id_house", "Объект", "id_zk","Долгота","Широта","Этап"]]
#
#     df1 = df1.groupby("id_house").agg({
#         "Объект": "first",
#         "id_zk": "first",
# }).reset_index().sort_values(["id_house", "id_zk"])
    df1 = df1.sort_values(["id_house", "id_zk"])
    if database.empty:
        df1.to_csv("base/bd.csv", index=False, sep=';', encoding='cp1251')
        database = df1.copy()
    else:
        # df2 = pd.read_csv("bd.csv", sep=';', encoding='cp1251')
        result = pd.merge(database, df1, how='outer', on=["id_house", "Объект", "id_zk","Долгота","Широта"],suffixes=('_x', ''))
        cols = result.columns.tolist()
        result = result.groupby("id_house").agg({
        "Объект": "first",
        "id_zk":"first",
        "Долгота": "sum",
        "Широта": "sum",
        "Этап_x": "last",
        "Этап":"first"
    }).reset_index().sort_values(["id_house"])
        for index,row in result.iterrows():
            if row["Этап"]=="":
                row["Этап"] = row["Этап_x"]
        for i in cols:
            if "x" in i:
                result.drop(i, axis="columns", inplace=True)
        result = result.sort_values(["id_house"])
        result.to_csv("base/bd.csv", index=False, sep=';', encoding='cp1251')
        database = result.copy()
    print(database)

def create_id_col_table(df,lst_col):
    global dict_id_zk
    dict_id_zk = {}
    count =0
    for i in df[lst_col[-1]]:
        if i not in dict_id_zk:
            dict_id_zk[i] = count
            count += 1
    df[lst_col[0]] = [dict_id_zk.get(i) for i in df[lst_col[-1]]]
    return df

def database_open():
    '''Присваивание переменной database таблицу с id ЖК и Дома '''
    global dict_id_zk,database
    try:
        database = pd.read_csv("base/bd.csv", sep=';', encoding='cp1251')
        db = database.groupby("Объект").agg({
            "id_zk": "first"
        }).reset_index().sort_values(["id_zk"])
        for index, row in db.iterrows():
            if row["Объект"] not in dict_id_zk:
                dict_id_zk[row["Объект"]] = row["id_zk"]
    except:
        database = pd.DataFrame(columns=["id", "Объект", "id_zk"])

def assignment_id(df):
    global database,df_type
    '''Перебор всех строк и присваивание им id ЖК и Дома,
    если же появляются новые значения, то присваивается новый id
    и позже происходит запись в основную bd'''
    new_house_zk=False
    # df["Количество проданных, кв.м."] = (df["Кол-во квартир по проектным декларациям, шт."]
    #                                      - df["Количество в остатках, шт."]) * df["Площадь, кв.м."]
    df["Количество проданных квартир, шт."] = (df["Кол-во квартир по проектным декларациям, шт."]
                                         - df["Количество в остатках, шт."])
    df["Площадь проданных квартир, кв.м."] = df["Количество проданных квартир, шт."]* df["Площадь, кв.м."]
    df = df.rename(columns={
        'id': 'id_house',
    })
    # df_type = df.groupby("Тип квартир").agg({
    #     "Площадь, кв.м.": "mean",
    # }).reset_index()
    # for index,row in df.iterrows():
    #     if math.isnan(row["Площадь, кв.м."]):
    #         try:
    #             df.loc[index, "Площадь, кв.м."] = df_type.loc[df_type["Тип квартир"]==row["Тип квартир"]]["Площадь, кв.м."].values[0]
    #         except:
    #                 pass

    df['Тип квартир'] = df['Тип квартир'].fillna("-")
    df.to_csv("her/" + name.split("/")[-1], index=False, sep=';', encoding='cp1251')
    df1 = df[["id_house", "Объект",
              "Тип квартир", "Средневз. стоимость квартиры, руб.",
              "Площадь, кв.м.","Цена за кв.м., руб./кв.м.","Кол-во квартир по проектным декларациям, шт.",
              "Количество в остатках, шт."]]
    df1 = df1.groupby("id_house").agg({
        "Объект": "first",
        "Тип квартир": lambda x: len(list(x)),
        "Средневз. стоимость квартиры, руб.": "sum",
        "Площадь, кв.м.": "sum",
        "Цена за кв.м., руб./кв.м.": "mean",
        "Кол-во квартир по проектным декларациям, шт.":"sum",
        "Количество в остатках, шт.":"sum"
    }).reset_index().sort_values(["id_house"])
    df1["Количество проданных квартир, шт."] = (df1["Кол-во квартир по проектным декларациям, шт."]
                                               - df1["Количество в остатках, шт."])
    df1["Площадь проданных квартир, кв.м."] = df1["Количество проданных квартир, шт."] * df1["Площадь, кв.м."]
    if database.empty:
        df1 = info_pars(df1,name)
        create_bd_id(df1)
    df_list_house = database.groupby("id_zk").agg({
        'id_house': lambda x: sorted(list(x)),
        "Объект": "first"
    }).reset_index().sort_values("id_zk")
    items = df_list_house.to_dict('records')
    df1["id_house"] = np.nan
    df1["id_zk"] = np.nan
    for item in items:
        count = 0
        for index, row in df1.iterrows():
            if row["Объект"] == item["Объект"] and len(item["id_house"]) > count:
                df1.loc[index, "id_house"] = item["id_house"][count]
                count += 1
            if row["Объект"] == item["Объект"]:
                row["id_zk"] = item["id_zk"]
    df1 = df1.sort_values(["id_house"])
    count = 1
    for index, row in df1.iterrows():
        if math.isnan(row["id_house"]):
            df1.loc[index, "id_house"] = database['id_house'].max() + count
            count += 1
            new_house_zk = True
    df1 = df1.sort_values(["id_zk"])
    for i in df1["Объект"]:
        if i not in dict_id_zk:
            dict_id_zk[i] = len(dict_id_zk)
            new_house_zk = True
    for index, row in df1.iterrows():
        df1.loc[index, "id_zk"] = dict_id_zk.get(row["Объект"])
    # df1.to_csv(name.split("/")[0] +"-sell1/" + name.split("/")[-1], index=False, sep=';', encoding='cp1251')
    # df1 = info_pars(df1, name)
    df1.to_csv(name.split("/")[0] +"-sell2/" + name.split("/")[-1], index=False, sep=';', encoding='cp1251')
    df1 = info_pars(df1, name)
    if new_house_zk:
        create_bd_id(df1)
    return df1

def fill_df_main_price(df):
    '''Перезаполнение id изначальной таблицы'''
    global df_sell
    # df1 = assignment_id(df)
    # iteration_df(df)
    df_sell = df_sell.sort_values(["id_house"])
    items = df_sell.to_dict('records')
    count = 0
    for index, row in df.iterrows():
        for item in items:
            if row["Объект"] == item["Объект"] and item["Тип квартир"] > 0:
                df.loc[index, "id_house"] = item["id_house"]
                item["Тип квартир"] = item["Тип квартир"] - 1
                if item["Тип квартир"] == 0:
                    items.remove(item)
                count += 1
                break
    df = fill_kvar_etap(df)
    return df.sort_values(["id_house"])

def add_column_to_bd_sell(df_main,df):
    '''Добавление столбца продаж к оснвной таблице'''
    items=df.to_dict('records')
    df_main[df.columns[-1]]=np.nan
    col=df.columns.tolist()
    for index, row in df_main.iterrows():
        for item in items:
            if row["Объект"] == item["Объект"] and row["Тип квартир"] == item["Тип квартир"] and row["id_house"]==item["id_house"]:
                df_main.loc[index, df.columns[-1]] = item[df.columns[-1]]
                items.remove(item)
                break
    df = pd.DataFrame(items)
    if not df.empty:
        result = pd.merge(df_main, df, how='outer', on=col)
        df_main = result.copy()
    df_main=df_main.sort_values(["id_house"])
    return df_main

def write_to_folder(df,folder):
    try:
        os.mkdir(name.split("/")[0] + folder.split("/")[0])
    except:
        pass
    df.to_csv(name.split("/")[0] + folder + name.split("/")[-1], index=False, sep=';', encoding='cp1251')

def assignment_sell_zk_house(df_main,col):
    '''Создание бд ЖК и ДОМА'''
    global df_sell
    id=base_name[verb_to_str(df_main)].get("list")[0]
    cols = base_name[verb_to_str(df_main)].get("list")
    df = df_sell[id.split(".")+[ "Объект", "Средневз. стоимость квартиры, руб.",
              "Площадь, кв.м.","Цена за кв.м., руб./кв.м.","Кол-во квартир по проектным декларациям, шт.","Количество проданных квартир, шт.",
              "Площадь проданных квартир, кв.м."]]
    df = df.groupby(id).agg({
        "Объект": "first",
        "Средневз. стоимость квартиры, руб.": "sum",
        "Площадь, кв.м.": "sum",
        "Цена за кв.м., руб./кв.м.": "mean",
        "Кол-во квартир по проектным декларациям, шт.":"sum",
        "Количество проданных квартир, шт.":"sum",
        "Площадь проданных квартир, кв.м.": "sum"
    }).reset_index().sort_values(id)
    df["Средневз. цена, руб."] = df["Средневз. стоимость квартиры, руб."] / df["Площадь, кв.м."]
    df_main = add_zk_house(df_main,df,col)

    # try:
    #     os.mkdir(name.split("/")[0] + folder.split("/")[0])
    # except:
    #     pass
    # df.to_csv(name.split("/")[0] + folder + name.split("/")[-1], index=False, sep=';', encoding='cp1251')
    return df_main

def iteration_df(df):
    '''Запись в бд Продаж'''
    global df_main_sell,df_zk_sell,df_house_sell,df_zk_price,df_house_price
    df = df.rename(columns={
        # 'Название': 'Объект',
        'Площадь проданных квартир, кв.м.': date,
    })
    df_col = df[["id_house", 'Объект', 'Тип квартир', date]]
    df_col = df_col.astype(object).replace({'—': np.nan, '-': np.nan})
    df_col['Тип квартир'] = df_col['Тип квартир'].fillna("-")
    df_col.iloc[:, 3] = pd.to_numeric(df_col.iloc[:, 3])
    if df_main_sell.empty:
        df_main_sell = df_col.sort_values(["id_house"]).copy()
    elif not df_main_sell.empty:
        df_main_sell = add_column_to_bd_sell(df_main_sell,df_col)
    df_zk_sell = assignment_sell_zk_house(df_zk_sell,"Площадь проданных квартир, кв.м.")
    df_house_sell = assignment_sell_zk_house(df_house_sell,"Площадь проданных квартир, кв.м.")
    df_zk_price = assignment_sell_zk_house(df_zk_price, "Цена за кв.м., руб./кв.м.")
    df_house_price = assignment_sell_zk_house(df_house_price, "Цена за кв.м., руб./кв.м.")
    # write_to_folder(df_zk_sell,"ZK-sell/")
    # write_to_folder(df_house_sell,"House-sell/")
    # write_to_folder(df_zk_price,"ZK-price/")
    # write_to_folder(df_house_price,"House-price/")


def info_pars(df1,name):
    lst_name = name.split('.')[0].split('/')[-1].split("-")
    lst_name.remove("data")
    name = "-".join(lst_name)
    df = pd.read_csv("info_csv/"+name+".csv", sep=';', encoding='cp1251')
    df = df.rename(columns={
        "id":"id_house",
        'Название': 'Объект'
    })
    df = df.astype(object).replace({'—': np.nan, '-': np.nan})
    df.loc[:, "Кол-во квартир (по проектным декларациям), шт."] = df.loc[:, "Кол-во квартир (по проектным декларациям), шт."].apply(pd.to_numeric)
    df.iloc[:,1:2] =df.iloc[:,1:2].apply(pd.to_numeric)
    # df = df[["id_house","Объект","Кол-во квартир (по проектным декларациям), шт."]]
    df1 = df1.sort_values(["id_house"])
    df1["Этап"]=np.nan
    df1["Долгота"] = np.nan
    df1["Широта"] = np.nan
    # print(len(df_sell),len(df))
    items = df1.to_dict('records')
    print(len(df1),len(df))
    print(items)
    for index, row in df.iterrows():
        for item in items:
            #and row["Кол-во квартир (по проектным декларациям), шт."] == item["Кол-во квартир по проектным декларациям, шт."]
            if row["Объект"] == item["Объект"]:
                # print(item["Объект"], item["id_house"])
                # print(index)
                df.loc[index, "id_house"] = item["id_house"]
                df1.loc[index,"Этап"] = row["Этап"]
                df1.loc[index, "Долгота"] = row["Долгота"]
                df1.loc[index, "Широта"] = row["Широта"]
                items.remove(item)
                break
    print(items)
    df = df.sort_values(["id_house"])
    df.to_csv("info_csv/"+name+".csv", index=False, sep=';', encoding='cp1251')
    return df1


def date_verb(name):
    '''Создание переменной date для переименования столбцов'''
    global date
    date = name.split('.')[0].split('/')[-1].split("-")
    date = datetime.datetime(int(date[0]), int(date[1]), int(date[2]))

def df_correct(name):
    '''Форматирование таблицы'''
    df = pd.read_csv(name, sep=';', encoding='cp1251')
    df = df.rename(columns={
        "id":"id_house",
        'Название': 'Объект'
    })
    df = df.astype(object).replace({'—': np.nan, '-': np.nan})
    df.iloc[:, 4:] = df.iloc[:, 4:].apply(pd.to_numeric)
    return df


#Перебор файлов в папке
def pars_folder_csv(path):
    '''Перебор файлов из входящей папки'''
    global df_sell,name
    dirs = os.listdir(path)
    lst_csv = sorted(dirs, key=lambda x: str(x.split(".")[0]))
    for file in lst_csv:
        name = path + "/" + file
        database_open()
        date_verb(name)
        if "ok" not in file:
            df_main = df_correct(name)
            df_sell = assignment_id(df_main)
            # info_pars(name)
            df_main = fill_df_main_price(df_main)
            add_to_df_main_price(df_main)
            iteration_df(df_main)
            df_main.to_csv(path + "/" + file.split(".")[0] + "-ok.csv", index=False, sep=';', encoding='cp1251')
            os.remove(path + "/" + file)
    record_base_excel_month()
            # df1 = create_new_date(name + "/" + file)


# Класс команд для управления файлом
class MyPrompt(Cmd):
    prompt = 'pb>'
    intro = 'Welcome! Type ? to list commands'

    def do_exit(self, inp):
        print('Bye')
        return True

    def help_exit(self):
        print('exit the application. Shorthand: x q Ctrl-D')

    def do_csv_to_excel(self, inp):
        csv_to_excel(inp)

    def help_csv_to_excel(self):
        print("Перевод значений из csv-файла в Excel")

    def do_excel_to_csv(self, inp):
        excel_to_csv(inp)

    def help_excel_to_csv(self):
        print("Перевод значений из Excel в csv")

    def do_run(self, inp):
        pars_folder_csv('price_csv')

    def help_run(self):
        print('''1)Добавить csv-файл или папку csv файлов к таблице цены Декарт/n
                2) Добавить в таблицу столбец Проданных шт и кв. м.
                3) Разделяет файл на 2 базы данных для Домов и для ЖК по ценам и продажам''')

    def default(self, inp):
        pass

if __name__ == '__main__':
    MyPrompt().cmdloop()
