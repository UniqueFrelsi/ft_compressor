from modules.compression.compressing import Image as MyImage
from modules.algorithms.algorithms import del_zeros
from modules.algorithms.haar import FHT
from modules.algorithms.walsh import FWT
import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import Image, ImageTk
import cv2
import numpy as np
from datetime import datetime


class Window:
    def __init__(self, width=800, height=560, title="Сжатие изображения"):
        self.in_path = ""
        self.out_path = ""
        self.FT = FHT()
        self.from_m = "BGR"
        self.to_m = "BGR"
        self.percent = [0, 0, 0]
        self.img = np.array([])
        self.mse = [0, 0, 0]
        self.psnr = 0

        self.root = tk.Tk()
        self.width = width
        self.height = height
        self.root.title(title)

        self.option_t = tk.IntVar(value=0)

        self.option_p = tk.IntVar(value=0)
        self.color1 = tk.StringVar(value="B")
        self.color2 = tk.StringVar(value="G")
        self.color3 = tk.StringVar(value="R")

        self.percent0 = tk.IntVar(value=0)
        self.percent1 = tk.IntVar(value=0)
        self.percent2 = tk.IntVar(value=0)

        self.step = tk.IntVar(value=3)

        self.load = tk.StringVar(value="Загрузка: 0:00:00.000000")
        self.start = tk.StringVar(value="Прямое: 0:00:00.000000")
        self.save = tk.StringVar(value="Обнуление: 0:00:00.000000")
        self.end = tk.StringVar(value="Обратное: 0:00:00.000000")
        self.full = tk.StringVar(value="Все время: 0:00:00.000000")

        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.settings = tk.Frame(self.root)
        self.settings.grid(row=0, column=0, sticky="nesw")
        self.result = tk.Frame(self.root)
        self.result.grid(row=0, column=1, sticky="nesw")

        self.l_c = tk.LabelFrame(self.settings, text="Цвета", width=90, height=117)
        self.l_c_1 = tk.Label(self.l_c, textvariable=self.color1)
        self.l_c_2 = tk.Label(self.l_c, textvariable=self.color2)
        self.l_c_3 = tk.Label(self.l_c, textvariable=self.color3)

        self.l_r = tk.LabelFrame(self.settings, text="Вывод", width=90, height=117)
        self.l_r_1 = tk.Label(self.l_r, textvariable=self.load)
        self.l_r_2 = tk.Label(self.l_r, textvariable=self.start)
        self.l_r_3 = tk.Label(self.l_r, textvariable=self.save)
        self.l_r_4 = tk.Label(self.l_r, textvariable=self.end)
        self.l_r_5 = tk.Label(self.l_r, textvariable=self.full)

    def run(self):
        self.draw_widgets()
        self.root.mainloop()

    def _open(self):
        self.in_path = askopenfilename(filetypes=[("Пользовательские файлы", "*.jpg *.jpeg *.png")])
        try:
            self.index = self.in_path[::-1].index(".") + 1
            self.extension = self.in_path[-self.index::]
        except ValueError:
            pass

    def _save(self):
        if self.extension == ".jpg" or self.extension == ".jpeg":
            types = [("JPEG Image", "*.jpg *.jpeg"), ("PNG Image", "*.png")]
        else:
            types = [("PNG Image", "*.png"), ("JPEG Image", "*.jpg *.jpeg")]
        self.out_path = asksaveasfilename(filetypes=types)
        self.get_path_save()

    def get_path_save(self):
        try:
            self.out_path[::-1].index(".") + 1
        except ValueError:
            if self.out_path != "":
                self.out_path = self.out_path + self.extension
            else:
                if self.in_path != "":
                    self.out_path = self.in_path[:-self.index] + "-com" + self.extension
                else:
                    pass

    def draw_menu(self):
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)
        file_menu = tk.Menu(menu_bar, tearoff=0)

        menu_bar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Открыть", command=self._open)
        file_menu.add_command(label="Сохранить как", command=self._save)

    def set_FHT(self):
        self.FT = FHT()

    def set_FWT(self):
        self.FT = FWT()

    def set_BGR(self):
        self.to_m = "BGR"
        self.color1.set("B")
        self.color2.set("G")
        self.color3.set("R")

    def set_YCrCb(self):
        self.to_m = "YCrCb"
        self.color1.set("Y")
        self.color2.set("Cr")
        self.color3.set("Cb")

    def clear_frame(self):
        for widget in self.result.winfo_children():
            widget.destroy()
        self.result.pack_forget()

    def compress(self):

        s = datetime.now().astimezone()
        c_0 = s
        self.clear_frame()
        self.get_path_save()

        self.percent = [self.percent0.get(), self.percent1.get(), self.percent2.get()]

        self.img = MyImage.get_image(self.in_path)
        self.img = MyImage.set_color_model(self.img, from_m=self.from_m, to_m=self.to_m)
        c = datetime.now().astimezone()
        self.load.set("Загрузка: {0}".format(c-c_0))

        if self.step.get() == 0:
            img_com = np.copy(self.img)
        else:
            img_com = self.FT.t_2d(self.img, self.step.get())
            c_0 = datetime.now().astimezone()
            self.start.set("Прямое: {0}".format(c_0 - c))

            img_com = MyImage.get_approx(img_com, self.percent, self.step.get())
            c = datetime.now().astimezone()
            self.save.set("Обнуление: {0}".format(c - c_0))

            img_com = self.FT.it_2d(img_com, self.step.get())
            c_0 = datetime.now().astimezone()
            img_com = del_zeros(img_com, self.img.shape)
            self.end.set("Обратное: {0}".format(c_0 - c))

        self.img = MyImage.set_color_model(self.img, from_m=self.to_m, to_m=self.from_m)
        img_com = MyImage.set_color_model(img_com, from_m=self.to_m, to_m=self.from_m)

        MyImage.save_image(img_com, self.out_path)
        c = datetime.now().astimezone()
        self.mse, self.psnr = MyImage.get_error(self.img, img_com)
        self.full.set("Все время: {0}".format(c - s))

        self.draw_result()

    def draw_settings(self):
        l_t = tk.LabelFrame(self.settings, text="Преобразование", width=90, height=70)
        l_t.pack(padx=12, pady=12, anchor=tk.SW, fill="both")
        tk.Radiobutton(l_t, text="FHT", variable=self.option_t, value=0, command=self.set_FHT).pack(side=tk.LEFT)
        tk.Radiobutton(l_t, text="FWT", variable=self.option_t, value=1, command=self.set_FWT).pack(side=tk.LEFT)

        l_n = tk.LabelFrame(self.settings, text="Размер квадратов", width=90, height=52)
        l_n.pack(padx=12, pady=12, anchor=tk.SW, fill="both")
        tk.Label(l_n, text="n =").place(x=30, y=4)
        tk.Spinbox(l_n, from_=-1, to=100, textvariable=self.step, width=10).place(x=60, y=5)

        l_p = tk.LabelFrame(self.settings, text="Палитра", width=90, height=60)
        l_p.pack(padx=12, pady=12, anchor=tk.SW, fill="both")
        tk.Radiobutton(l_p, text="BGR", variable=self.option_p, value=0, command=self.set_BGR).pack(side=tk.LEFT)
        tk.Radiobutton(l_p, text="YCrCb", variable=self.option_p, value=1, command=self.set_YCrCb).pack(side=tk.LEFT)
        self.l_c.pack(padx=12, pady=12, anchor=tk.SW, fill="both")
        self.l_c_1.place(x=3, y=5)
        tk.Label(self.l_c, text="0 percent").place(x=25, y=5)
        tk.Spinbox(self.l_c, from_=0, to=100, textvariable=self.percent0, width=8).place(x=80, y=5)
        self.l_c_2.place(x=3, y=35)
        tk.Label(self.l_c, text="0 percent").place(x=25, y=35)
        tk.Spinbox(self.l_c, from_=0, to=100, textvariable=self.percent1, width=8).place(x=80, y=35)
        self.l_c_3.place(x=3, y=65)
        tk.Label(self.l_c, text="0 percent").place(x=25, y=65)
        tk.Spinbox(self.l_c, from_=0, to=100, textvariable=self.percent2, width=8).place(x=80, y=65)

        self.l_r.pack(padx=12, pady=12, anchor=tk.SW, fill="both")
        self.l_r_1.pack(anchor=tk.SW)
        self.l_r_2.pack(anchor=tk.SW)
        self.l_r_3.pack(anchor=tk.SW)
        self.l_r_4.pack(anchor=tk.SW)
        self.l_r_5.pack(anchor=tk.SW)

        tk.Button(self.settings, text="Выполнить", command=self.compress).place(x=90, y=500)

    def draw_result(self):
        l_i = tk.LabelFrame(self.result, text="До сжатия/После сжатия")

        l_i.pack(anchor=tk.NW, fill="both")
        b, g, r = cv2.split(self.img)
        img = cv2.merge((r, g, b))
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img)
        label_im1 = tk.Label(l_i, image=img_tk)
        label_im1.image = img_tk
        label_im1.pack(side=tk.LEFT, anchor=tk.NW)

        img_com = MyImage.get_image(self.out_path)
        b, g, r = cv2.split(img_com)
        img_com = cv2.merge((r, g, b))
        img_com = Image.fromarray(img_com)
        img_tk_com = ImageTk.PhotoImage(image=img_com)
        label_im2 = tk.Label(l_i, image=img_tk_com)
        label_im2.image = img_tk_com
        label_im2.pack(side=tk.LEFT, anchor=tk.NW)

        l_e = tk.LabelFrame(self.result, text="Статистика", width=90, height=60)
        l_e.pack(padx=12, pady=12, anchor=tk.NW, fill="both")
        tk.Label(l_e, text="MSE: B - {0}, G - {1}, R - {2}".format(self.mse[0], self.mse[1], self.mse[2])).pack(expand=1)
        tk.Label(l_e, text="PSNR: {0}".format(self.psnr)).pack(expand=1)

    def draw_widgets(self):
        self.root.minsize(width=self.width, height=self.height)
        self.draw_menu()
        self.draw_settings()
