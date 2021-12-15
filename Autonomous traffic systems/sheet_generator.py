import xlrd
#python data to sheet
import xlwt
from xlwt import Workbook
import datetime
#plotting the trajectory and bounding box
from matplotlib import image
from matplotlib import pyplot as plt
import time
import numpy as np



#id_no = 5                          ###########input id no (manually or automated later)
Fps = 30

### loading file
loc = ("date 19-8.xls")
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)

''' Single file generator
# Workbook is created
wb1 = Workbook()
# add_sheet is used to create sheet.
sheet1 = wb1.add_sheet("sheet 1", cell_overwrite_ok=True)
sheet1.write(0, 1, 'id')
sheet1.write(0, 2, 'frame_no')
sheet1.write(0, 3, 'x')
sheet1.write(0, 4, 'y')
sheet1.write(0, 5, 'type and confidence')
sheet1.write(0, 6, 'x_real')
sheet1.write(0, 7, 'y_real')
sheet1.write(0, 10, 'Fps')
sheet1.write(0, 11, 'time')

j = 2
## nrows = no of rows
#print(sheet.nrows, type(sheet.cell_value(3, 1)))
#for i in range(1, int(sheet.nrows)):

for i in range(2, 10):
    if int(sheet.cell_value(i, 1)) == id_no:
        #print("hii")
        sheet1.write(j, 1, id_no)
        sheet1.write(j, 2, sheet.cell_value(i, 2))
        sheet1.write(j, 3, sheet.cell_value(i, 3))
        sheet1.write(j, 4, sheet.cell_value(i, 4))
        sheet1.write(j, 5, sheet.cell_value(i, 5))
        sheet1.write(j, 6, sheet.cell_value(i, 6))
        sheet1.write(j, 7, sheet.cell_value(i, 7))
        sheet1.write(j, 10, Fps)
        j = j + 1

print(j)
print(i)


wb1.save('generated output1.xls')

'''

for i in range(0, 10):
    # Workbook is created
    wb1 = Workbook()
    # add_sheet is used to create sheet.
    sheet1 = wb1.add_sheet("sheet 1", cell_overwrite_ok=True)
    sheet1.write(0, 1, 'id')
    sheet1.write(0, 2, 'frame_no')
    sheet1.write(0, 3, 'x')
    sheet1.write(0, 4, 'y')
    sheet1.write(0, 5, 'type and confidence')
    sheet1.write(0, 6, 'x_real')
    sheet1.write(0, 7, 'y_real')
    sheet1.write(0, 8, 'Fps')
    sheet1.write(0, 9, 'time')
    sheet1.write(0, 10, 'time_cu')
    sheet1.write(0, 11, 'delta_Y')
    sheet1.write(0, 12, 'insta velo')
    sheet1.write(0, 13, 'Avg velo')


    id_no = i
    name_id = -1
    yp = 0
    yn = 0
    Fp = 0
    Fn = 0
    j = 2
    time_cu = 1e-8
    for k in range(2, int(sheet.nrows)):
        if int(sheet.cell_value(k, 1)) == id_no:
            name_id = id_no
            #print(name_id)
            if j == 2:
                sheet1.write(j, 1, id_no)
                sheet1.write(j, 2, sheet.cell_value(k, 2))
                sheet1.write(j, 3, sheet.cell_value(k, 3))
                sheet1.write(j, 4, sheet.cell_value(k, 4))
                sheet1.write(j, 5, sheet.cell_value(k, 5))
                sheet1.write(j, 6, sheet.cell_value(k, 6))
                sheet1.write(j, 7, sheet.cell_value(k, 7))
                Y_B = sheet.cell_value(k, 7)
                yp = sheet.cell_value(k, 7)
                Fp = sheet.cell_value(k, 2)
                sheet1.write(j, 8, Fps)
                sheet1.write(j, 9, 0)
                sheet1.write(j, 10, time_cu)
                sheet1.write(j, 11, 0)
                sheet1.write(j, 12, 0)
                sheet1.write(j, 13, 0)
                j = j + 1
                print("done")
            else:
                sheet1.write(j, 1, id_no)
                sheet1.write(j, 2, sheet.cell_value(k, 2))
                sheet1.write(j, 3, sheet.cell_value(k, 3))
                sheet1.write(j, 4, sheet.cell_value(k, 4))
                sheet1.write(j, 5, sheet.cell_value(k, 5))
                sheet1.write(j, 6, sheet.cell_value(k, 6))
                sheet1.write(j, 7, sheet.cell_value(k, 7))
                sheet1.write(j, 8, Fps)
                ### time F2-F1 /Fps
                yn = sheet.cell_value(k, 7)
                Fn = sheet.cell_value(k, 2)

                time = Fn - Fp
                #print(time)
                time = float(time/Fps)
                #print("time",time)

                ### time cummulative
                time_cu = time_cu + time
                #print(time_cu)

                ############### delta Y
                delta_Y = yn - yp
                ############## insta velo
                if time != 0:
                    insta_velo = float((delta_Y*18)/(time*5))
                else:
                    insta_velo = 0
                ########### avg velo

                avg_velo = yn - Y_B
                #print(avg_velo)
                avg_velo = float(avg_velo*18/(time_cu*5))
                #print("y1", delta_Y)
                #print("time_cu", time_cu)
                #print("insta_velo", insta_velo)
                #print(avg_velo)
                sheet1.write(j, 9, time)
                sheet1.write(j, 10, time_cu)
                sheet1.write(j, 11, delta_Y)
                sheet1.write(j, 12, insta_velo)
                sheet1.write(j, 13, avg_velo)
                Fp = Fn
                yp = yn
                j = j + 1

    if name_id != -1:
        save_name = 'sheets/generated output' + str(name_id)
        wb1.save(save_name + '.xls')
print("done complete")
