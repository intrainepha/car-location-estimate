package main

import (
	"encoding/xml"
	"flag"
	"image/png"
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
	Y string `xml:"y"`
	X string `xml:"x"`
}

func checkError(e error) {
	if e != nil {
		log.Fatal(e)
	}
}

func contains(ss []string, str string) bool {
	/*Check if a string exists in a slice of string

	Args:
		str(string): string data

	Returns:
		(bool): Check result
	*/

	for _, v := range ss {
		if v == str {
			return true
		}
	}

	return false
}

func str2int(str string) int {
	/*Convert string to int

	Args:
		str(string): string data

	Returns:
		floatNum(float64): int data
	*/

	intNum, err := strconv.Atoi(str)
	checkError(err)

	return intNum
}

func str2float64(str string) float64 {
	/*Convert string to float64

	Args:
		str(string): string data

	Returns:
		intNum(float64): float64 data
	*/

	floatNum, err := strconv.ParseFloat(str, 64)
	checkError(err)

	return floatNum
}

func crop(root string, freq int, cls []string) error {
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
	checkError(err)
	imgDir := path.Join(root, "images")
	labelDir := path.Join(root, "labels")
	// roiImgDir := path.Join(root, "roi_img")
	// roiXmlDir := path.Join(root, "roi_xml")
	// roiTxtDir := path.Join(root, "roi_txt")
	files, err := os.ReadDir(imgDir)
	checkError(err)
	for i, f := range files {
		if i%freq != 0 {
			continue
		}
		imgFile, err := os.Open(path.Join(imgDir, f.Name()))
		checkError(err)
		defer imgFile.Close()
		imgData, err := png.DecodeConfig(imgFile)
		imgW, imgH := imgData.Width, imgData.Height
		checkError(err)
		bytes, err := os.ReadFile(path.Join(labelDir, strings.Replace(f.Name(), ".png", ".txt", 1)))
		checkError(err)
		for _, l := range strings.Split(string(bytes), "\n") {
			if l == "" {
				continue
			}
			info := strings.Split(l, " ")
			// info[1]: float from 0(non-truncated) to 1(truncated)
			truncated := str2float64(info[1])
			// info[2]: 0->fully visible, 1->partial occluded, 2->largely occluded, 3=unknown
			occluded := str2int(info[2])
			locX, locY := str2float64(info[11]), str2float64(info[13])
			// Filter with X range=[-8, 8], Y range=[0, 80]
			if truncated != 0 || occluded != 0 || !contains(cls, info[0]) || math.Abs(locX) > 8 || locY < 0 || locY > 80 {
				continue
			}
			xmin, xmax := str2float64(info[4]), str2float64(info[5])
			ymin, ymax := str2float64(info[6]), str2float64(info[7])
			w, h := xmax-xmin, ymax-ymin
			offsetX, offsetY := w*0.25, h*0.25
			roiXmin, roiXmax := int(xmin-offsetX), int(xmax+offsetX)
			roiYmin, roiYmax := int(ymin-offsetY), int(ymax+offsetY)
			if roiXmin < 0 {
				roiXmin = 0
			}
			if roiYmin < 0 {
				roiYmin = 0
			}
			if roiXmax >= imgW {
				roiXmax = imgW - 1
			}
			if roiYmax >= imgH {
				roiYmax = imgH - 1
			}
			roiW, roiH := roiXmax-roiXmin+1, roiYmax-roiYmin+1
			// TODO:
			// 	1. Calculate bbox relative to ROI;
			// 	2. Write ROI image;
			// 	2. Write xml;
			// 	2. Write txt;
		}
	}

	return nil
}

func list(root string, cls []string) error {
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
	checkError(err)
	txt := path.Join(root, "paths.txt")
	file, err := os.OpenFile(txt, os.O_RDWR|os.O_CREATE, 0755)
	checkError(err)
	defer file.Close()
	for i, c := range cls {
		log.Println(i, c)
		imgDir := path.Join(root, c, "images")
		_, err := os.Stat(imgDir)
		checkError(err)
		files, err := os.ReadDir(imgDir)
		checkError(err)
		for _, f := range files {
			path := path.Join(imgDir, f.Name())
			file.WriteString(path + "\n")
		}

	}
	return nil
}

func genTxt(root string, cls []string) error {
	/*Read *.xml lable files and generate *.txt files.

	Args:
		root(string): Dataset directory
		cls([]string): classes you choose to generate

	Returns:
		error
	*/

	root, err := filepath.Abs(root)
	checkError(err)
	for id, c := range cls {
		xmlDir := path.Join(root, c, "annotations")
		txtDir := path.Join(root, c, "labels")
		xmls, err := os.ReadDir(xmlDir)
		checkError(err)
		for _, x := range xmls {
			xmlPath := path.Join(xmlDir, x.Name())
			bytes, _ := os.ReadFile(xmlPath)
			var data XMLData
			_ = xml.Unmarshal([]byte(bytes), &data)
			objXCtr := strconv.FormatFloat((data.Obj.Box.Xmin+data.Obj.Box.Xmax)/2/data.Roi.Size.W, 'g', -1, 64)
			objYCtr := strconv.FormatFloat((data.Obj.Box.Ymin+data.Obj.Box.Ymax)/2/data.Roi.Size.H, 'g', -1, 64)
			objW := strconv.FormatFloat((data.Obj.Box.Xmax-data.Obj.Box.Xmin+1)/data.Roi.Size.W, 'g', -1, 64)
			objH := strconv.FormatFloat((data.Obj.Box.Ymax-data.Obj.Box.Ymin+1)/data.Roi.Size.H, 'g', -1, 64)
			roiXCtr := strconv.FormatFloat((data.Roi.Box.Xmin+data.Roi.Box.Xmax)/2/data.Size.W, 'g', -1, 64)
			roiYCtr := strconv.FormatFloat((data.Roi.Box.Ymin+data.Roi.Box.Ymax)/2/data.Size.H, 'g', -1, 64)
			roiW := strconv.FormatFloat((data.Obj.Box.Xmax-data.Obj.Box.Xmin+1)/data.Size.W, 'g', -1, 64)
			roiH := strconv.FormatFloat((data.Obj.Box.Ymax-data.Obj.Box.Ymin+1)/data.Size.H, 'g', -1, 64)
			str := []string{
				strconv.Itoa(id),
				objXCtr, objYCtr, objW, objH,
				data.Obj.Loc.Y, data.Obj.Loc.X,
				roiXCtr, roiYCtr, roiW, roiH,
			}
			txtName := strings.Replace(x.Name(), ".xml", ".txt", 1)
			txtPath := path.Join(txtDir, txtName)
			buffer, err := os.OpenFile(txtPath, os.O_RDWR|os.O_CREATE, 0755)
			checkError(err)
			defer buffer.Close()
			buffer.WriteString(strings.Join(str, " "))
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
	cropCmd := flag.NewFlagSet("gt", flag.ExitOnError)
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
		err := list(*listDir, classes)
		checkError(err)
	case "gt":
		gtCmd.Parse(os.Args[2:])
		classes := strings.Split(*gtCls, ",")
		err := genTxt(*gtDir, classes)
		checkError(err)
	case "crop":
		cropCmd.Parse(os.Args[2:])
		classes := strings.Split(*cropCls, ",")
		err := crop(*cropDir, *cropFreq, classes)
		checkError(err)
	default:
		log.Println("Expected subcommands")
		os.Exit(1)
	}
}
