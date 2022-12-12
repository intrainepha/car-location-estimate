package main

import (
	"bytes"
	"encoding/xml"
	"flag"
	"image"
	"image/draw"
	"image/jpeg"
	_ "image/png"
	"log"
	"math"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
)

type XMLData struct {
	Size Size   `xml:"size"`
	Roi  ROI    `xml:"roi"`
	Obj  Object `xml:"obj"`
}

type Size struct {
	W float64 `xml:"w"`
	H float64 `xml:"h"`
}

type ROI struct {
	Size Size `xml:"size"`
	Box  Box  `xml:"bbox"`
}
type Box struct {
	Xmin float64 `xml:"xmin"`
	Ymin float64 `xml:"ymin"`
	Xmax float64 `xml:"xmax"`
	Ymax float64 `xml:"ymax"`
}
type Object struct {
	Name string   `xml:"name"`
	Box  Box      `xml:"bndbox"`
	Loc  Location `xml:"location"`
}

type Location struct {
	X string `xml:"x"`
	Y string `xml:"y"`
}

func checkE(e error) {
	if e != nil {
		log.Panic(e)
	}
}

func cleanDir(dir string) {
	err := os.RemoveAll(dir)
	checkE(err)
	err = os.MkdirAll(dir, 0755)
	checkE(err)
}

func getClassID(ss []string, str string) (int, bool) {
	/*Check if a string exists in a slice of string

	Args:
		str(string): string data

	Returns:
		(bool): Check result
	*/

	for i, v := range ss {
		if v == str {
			return i, true
		}
	}

	return -1, false
}

func str2int(str string) int {
	/*Convert string to int

	Args:
		str(string): string data

	Returns:
		floatNum(float64): int data
	*/

	intNum, err := strconv.Atoi(str)
	checkE(err)

	return intNum
}

func str2f64(str string) float64 {
	/*Convert string to float64

	Args:
		str(string): string data

	Returns:
		intNum(float64): float64 data
	*/

	floatNum, err := strconv.ParseFloat(str, 64)
	checkE(err)

	return floatNum
}

func f642Str(num float64) string {
	/*Convert float64 to string

	Args:
		num(float64): float64 data

	Returns:
		(string): string data
	*/

	return strconv.FormatFloat(num, 'g', -1, 64)
}

func loadImg(path string) (image.Image, [2]int) {
	/*Load image fron a file path

	Args:
		path(string): Path to image file

	Returns:
		(image.Image): Readed image data
		([2]int): Image size=[width, height]
	*/

	imgBytes, err := os.ReadFile(path)
	checkE(err)
	imgInfo, _, err := image.DecodeConfig(bytes.NewReader(imgBytes))
	checkE(err)
	size := [2]int{imgInfo.Width, imgInfo.Height}
	data, _, err := image.Decode(bytes.NewReader(imgBytes))
	checkE(err)

	return data, size
}

func saveImg(img image.Image, path string) {
	/*Save image in format=[png, jpg, gif]

	Args:
		img(image.Image): Readed image data
		path(image.Image): save path

	Returns:
		None
	*/

	formats := [...]string{"png", "jpg"}
	strs := strings.Split(path, ".")
	suffix := strs[len(strs)-1]
	var confirmed bool = false
	for _, f := range formats {
		if suffix == f {
			confirmed = true
			break
		}
	}
	if !confirmed {
		log.Panic("Unsupported format:", suffix)
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE, 0755)
	checkE(err)
	err = jpeg.Encode(f, img, &jpeg.Options{Quality: 100})
	checkE(err)
}

func cropImg(img image.Image, rect image.Rectangle) image.Image {
	/*Save image in format=[png, jpg, gif]

	Args:
		img(image.Image): Readed image data
		path(image.Image): save path

	Returns:
		None
	*/

	var res image.Image
	type subImageSupported interface {
		SubImage(r image.Rectangle) image.Image
	}
	if sImg, ok := img.(subImageSupported); ok {
		res = sImg.SubImage(rect)
	} else {
		res := image.NewRGBA(rect)
		draw.Draw(res, rect, img, rect.Min, draw.Src)
	}

	return res
}

func saveTXT(path string, imgSzie [2]int, id int, rbox Box, location [2]float64, obox Box) {
	/*Calculate data and write to *.txt files.

	Args:
		path(string): TXT file path
		cls([]string): classes you choose to generate

	Returns:
		error
	*/

	ow, oh := obox.Xmax-obox.Xmin+1, obox.Ymax-obox.Ymin+1
	rw, rh := rbox.Xmax-rbox.Xmin+1, rbox.Ymax-rbox.Ymin+1
	xCtrObj, yCtrObj := (obox.Xmin+obox.Xmax)/2/rw, (obox.Ymin+obox.Ymax)/2/rh
	wObj, hObj := ow/rw, oh/rh
	xCtrROI, yCtrROI := (rbox.Xmin+rbox.Xmax)/2/float64(imgSzie[0]), (rbox.Ymin+rbox.Ymax)/2/float64(imgSzie[1])
	wROI, hROI := rw/float64(imgSzie[0]), oh/float64(imgSzie[1])
	str := []string{
		strconv.Itoa(id),
		f642Str(xCtrObj), f642Str(yCtrObj), f642Str(wObj), f642Str(hObj),
		f642Str(location[1]), f642Str(location[0]),
		f642Str(xCtrROI), f642Str(yCtrROI), f642Str(wROI), f642Str(hROI),
	}
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE, 0755)
	checkE(err)
	defer file.Close()
	file.WriteString(strings.Join(str, " "))
}

func ruleBox(box Box, size [2]int) {
	/*Adjust box value taht out of boundary

	Args:
		box([4]float64): Box=[Xtl, Ytl, Xbr, Ybr]
		size([2]int): Boundary=[X-aix boundary, y-aix boundary]

	Returns:
		error
	*/

	if box.Xmin < 0 {
		box.Xmin = 0
	}
	if box.Ymin < 0 {
		box.Ymin = 0
	}
	if box.Xmax >= float64(size[0]) {
		box.Xmax = float64(size[1]) - 1
	}
	if box.Ymax >= float64(size[0]) {
		box.Ymax = float64(size[1]) - 1
	}
}

func runCrop(root string, freq int, cls []string) error {
	/*Crop Region of interest (ROI) from image with label formated in kitti approch.
			root/
				├──images
				|   └──*.png
				└──labels
				└──*.txt

	Args:
		root(string): Directory contains data files
		freq(int): Frequence for filtering images

	Returns:
		error
	*/

	root, err := filepath.Abs(root)
	checkE(err)
	imgDir := path.Join(root, "images")
	labelDir := path.Join(root, "labels")
	imgDirROI := path.Join(root, "roi_img")
	cleanDir(imgDirROI)
	labelDirROI := path.Join(root, "roi_txt")
	cleanDir(labelDirROI)
	files, err := os.ReadDir(imgDir)
	checkE(err)
	for i, f := range files {
		if i%freq != 0 {
			continue
		}
		imgData, imgSize := loadImg(path.Join(imgDir, f.Name()))
		bytesTXT, err := os.ReadFile(path.Join(labelDir, strings.Replace(f.Name(), ".png", ".txt", 1)))
		checkE(err)
		for j, l := range strings.Split(string(bytesTXT), "\n") {
			if l == "" {
				continue
			}
			info := strings.Split(l, " ")
			// info[1]: float from 0(non-truncated) to 1(truncated)
			truncated := str2f64(info[1])
			// info[2]: 0->fully visible, 1->partial occluded, 2->largely occluded, 3=unknown
			occluded := str2int(info[2])
			loc := [2]float64{str2f64(info[11]), str2f64(info[13])}
			// Filter with X range=[-8, 8], Y range=[0, 80]
			clsID, contained := getClassID(cls, info[0])
			if truncated != 0 || occluded != 0 || !contained || math.Abs(loc[0]) > 8 || loc[1] < 0 || loc[1] > 80 {
				continue
			}
			bbox := Box{str2f64(info[4]), str2f64(info[5]), str2f64(info[6]), str2f64(info[7])}
			bw, bh := bbox.Xmax-bbox.Xmin+1, bbox.Ymax-bbox.Ymin+1
			// fmt.Println(bbox, bw, bh)
			offsetX, offsetY := bw*0.25, bh*0.25
			// fmt.Println(offsetX, offsetY)
			rbox := Box{bbox.Xmin - offsetX, bbox.Ymin - offsetY, bbox.Xmax + offsetX, bbox.Ymax + offsetY}
			ruleBox(rbox, imgSize)
			// fmt.Println(rbox)
			// Calculate bbox relative to ROI
			obox := Box{bbox.Xmin - rbox.Xmin, bbox.Ymin - rbox.Ymin, bbox.Xmax - rbox.Xmin, bbox.Ymax - rbox.Ymin}
			rw, rh := rbox.Xmax-rbox.Xmin+1, rbox.Ymax-rbox.Ymin+1
			// fmt.Println(obox, rw, rh)
			if obox.Xmin < 0 || obox.Ymin < 0 || obox.Xmax > rw || obox.Xmax > rh {
				continue
			}
			// Write ROI image
			imgPathROI := path.Join(imgDirROI, strings.Replace(f.Name(), ".png", "_"+strconv.Itoa(j)+".jpg", 1))
			subImg := cropImg(imgData, image.Rect(int(rbox.Xmin), int(rbox.Ymin), int(rbox.Xmax), int(rbox.Ymax)))
			saveImg(subImg, imgPathROI)
			checkE(err)
			// Write txt
			labelPathROI := path.Join(labelDirROI, strings.Replace(f.Name(), ".png", "_"+strconv.Itoa(j)+".txt", 1))
			saveTXT(labelPathROI, imgSize, clsID, rbox, loc, obox)
			// debug netive box value
			// Augment
		}
	}

	return nil
}

func runList(root string, cls []string) error {
	/*Generate paths.txt file, which contains data paths with each
	data path in one line, this file format is required by yolo-v3.

	Args:
		root(string): Directory contains data files
			root/
				├──class_0
				|   ├──data_file_0
				|   ├── ...
				|   └──data_file_n
				├── ...
				└──class_n
					├──data_file_0
					├── ...
					└──data_file_n
		cls([]string): classes you choose to generate

	Returns:
		error
	*/

	root, err := filepath.Abs(root)
	log.Println("Target Directory:", root)
	checkE(err)
	txt := path.Join(root, "paths.txt")
	file, err := os.OpenFile(txt, os.O_RDONLY|os.O_CREATE, 0755)
	checkE(err)
	defer file.Close()
	for i, c := range cls {
		log.Println(i, c)
		imgDir := path.Join(root, c, "images")
		_, err := os.Stat(imgDir)
		checkE(err)
		files, err := os.ReadDir(imgDir)
		checkE(err)
		for _, f := range files {
			path := path.Join(imgDir, f.Name())
			file.WriteString(path + "\n")
		}

	}
	return nil
}

func genTXT(root string, cls []string) error {
	/*Read *.xml lable files and generate *.txt files.

	Args:
		root(string): Dataset directory
		cls([]string): classes you choose to generate

	Returns:
		error
	*/

	root, err := filepath.Abs(root)
	checkE(err)
	for id, c := range cls {
		xmlDir := path.Join(root, c, "annotations")
		txtDir := path.Join(root, c, "labels")
		xmls, err := os.ReadDir(xmlDir)
		checkE(err)
		for _, x := range xmls {
			xmlPath := path.Join(xmlDir, x.Name())
			bytes, _ := os.ReadFile(xmlPath)
			var data XMLData
			_ = xml.Unmarshal([]byte(bytes), &data)
			xCtrObj := (data.Obj.Box.Xmin + data.Obj.Box.Xmax) / 2 / data.Roi.Size.W
			yCtrObj := (data.Obj.Box.Ymin + data.Obj.Box.Ymax) / 2 / data.Roi.Size.H
			wObj := (data.Obj.Box.Xmax - data.Obj.Box.Xmin + 1) / data.Roi.Size.W
			hObj := (data.Obj.Box.Ymax - data.Obj.Box.Ymin + 1) / data.Roi.Size.H
			xCtrROI := (data.Roi.Box.Xmin + data.Roi.Box.Xmax) / 2 / data.Size.W
			yCtrROI := (data.Roi.Box.Ymin + data.Roi.Box.Ymax) / 2 / data.Size.H
			wROI := (data.Obj.Box.Xmax - data.Obj.Box.Xmin + 1) / data.Size.W
			hROI := (data.Obj.Box.Ymax - data.Obj.Box.Ymin + 1) / data.Size.H
			str := []string{
				strconv.Itoa(id),
				f642Str(xCtrObj), f642Str(yCtrObj), f642Str(wObj), f642Str(hObj),
				data.Obj.Loc.Y, data.Obj.Loc.X,
				f642Str(xCtrROI), f642Str(yCtrROI), f642Str(wROI), f642Str(hROI),
			}
			txtName := strings.Replace(x.Name(), ".xml", ".txt", 1)
			txtPath := path.Join(txtDir, txtName)
			file, err := os.OpenFile(txtPath, os.O_WRONLY|os.O_CREATE, 0755)
			checkE(err)
			defer file.Close()
			file.WriteString(strings.Join(str, " "))
		}
	}

	return nil
}

func main() {
	listCmd := flag.NewFlagSet("list", flag.ExitOnError)
	listDir := listCmd.String("dir", "", "Directory")
	listCls := listCmd.String("cls", "", "Classes")
	gtCmd := flag.NewFlagSet("gt", flag.ExitOnError)
	gtDir := gtCmd.String("dir", "", "Directory")
	gtCls := gtCmd.String("cls", "", "Classes")
	cropCmd := flag.NewFlagSet("crop", flag.ExitOnError)
	cropDir := cropCmd.String("dir", "", "Directory")
	cropFreq := cropCmd.Int("freq", 1, "Frequence")
	cropCls := cropCmd.String("cls", "", "Classes")
	if len(os.Args) < 2 {
		log.Fatal("Expected subcommands!")
	}
	switch os.Args[1] {
	case "list":
		listCmd.Parse(os.Args[2:])
		classes := strings.Split(*listCls, ",")
		err := runList(*listDir, classes)
		checkE(err)
	case "gt":
		gtCmd.Parse(os.Args[2:])
		classes := strings.Split(*gtCls, ",")
		err := genTXT(*gtDir, classes)
		checkE(err)
	case "crop":
		cropCmd.Parse(os.Args[2:])
		classes := strings.Split(*cropCls, ",")
		err := runCrop(*cropDir, *cropFreq, classes)
		checkE(err)
	default:
		log.Println("Expected subcommands")
		os.Exit(1)
	}
}
