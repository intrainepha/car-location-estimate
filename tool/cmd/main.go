package main

import (
	"flag"
	"image"
	"log"
	"math"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	file "github.com/intrainepha/car-location-estimation/tool/src/file"
	op "github.com/intrainepha/car-location-estimation/tool/src/ops"
	tp "github.com/intrainepha/car-location-estimation/tool/src/types"
)

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
	op.CheckE(err)
	imgDir := path.Join(root, "images")
	labelDir := path.Join(root, "labels")
	imgDirROI := path.Join(root, "roi_img")
	op.CleanDir(imgDirROI)
	labelDirROI := path.Join(root, "roi_txt")
	op.CleanDir(labelDirROI)
	files, err := os.ReadDir(imgDir)
	op.CheckE(err)
	for i, f := range files {
		if i%freq != 0 {
			continue
		}
		imgData, imgSize := file.LoadImg(path.Join(imgDir, f.Name()))
		bytesTXT, err := os.ReadFile(path.Join(labelDir, strings.Replace(f.Name(), ".png", ".txt", 1)))
		op.CheckE(err)
		for j, l := range strings.Split(string(bytesTXT), "\n") {
			if l == "" {
				continue
			}
			info := strings.Split(l, " ")
			// info[1]: float from 0(non-truncated) to 1(truncated)
			truncated := op.Str2f64(info[1])
			// info[2]: 0->fully visible, 1->partial occluded, 2->largely occluded, 3=unknown
			occluded := op.Str2int(info[2])
			loc := &tp.Location{X: op.Str2f64(info[11]), Y: op.Str2f64(info[13])}
			// Filter with X range=[-8, 8], Y range=[0, 80]
			clsID, contained := getClassID(cls, info[0])
			if truncated != 0 || occluded != 0 || !contained || math.Abs(loc.X) > 8 || loc.Y < 0 || loc.Y > 80 {
				continue
			}
			bRct := &tp.Rect{
				Xtl: op.Str2f64(info[4]), Ytl: op.Str2f64(info[5]),
				Xbr: op.Str2f64(info[6]), Ybr: op.Str2f64(info[7]),
			}
			bb := tp.NewBox(bRct, &tp.Size{W: imgSize.W, H: imgSize.H})
			offX, offY := bb.Sz.W*0.25, bb.Sz.H*0.25
			rRct := &tp.Rect{
				Xtl: bb.Rct.Xtl - offX, Ytl: bb.Rct.Ytl - offY,
				Xbr: bb.Rct.Xbr + offX, Ybr: bb.Rct.Ybr + offY,
			}
			rb := tp.NewBox(rRct, &tp.Size{W: imgSize.W, H: imgSize.H})
			// Calculate bbox relative to ROI
			oRct := &tp.Rect{
				Xtl: bb.Rct.Xtl - rb.Rct.Xtl, Ytl: bb.Rct.Ytl - rb.Rct.Ytl,
				Xbr: bb.Rct.Xbr - rb.Rct.Xtl, Ybr: bb.Rct.Ybr - rb.Rct.Ytl,
			}
			ob := tp.NewBox(oRct, &tp.Size{W: rb.Sz.W, H: rb.Sz.H})
			// Write ROI image
			imgPathROI := path.Join(imgDirROI, strings.Replace(f.Name(), ".png", "_"+strconv.Itoa(j)+".jpg", 1))
			subImg := file.CropImg(
				imgData, image.Rect(int(rb.Rct.Xtl), int(rb.Rct.Ytl), int(rb.Rct.Xbr), int(rb.Rct.Ybr)),
			)
			file.SaveImg(subImg, imgPathROI)
			// Write txt
			labelPathROI := path.Join(labelDirROI, strings.Replace(f.Name(), ".png", "_"+strconv.Itoa(j)+".txt", 1))
			file.SaveTXT(clsID, labelPathROI, ob, loc, rb)
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
	op.CheckE(err)
	txt := path.Join(root, "paths.txt")
	file, err := os.OpenFile(txt, os.O_RDONLY|os.O_CREATE, 0755)
	op.CheckE(err)
	defer file.Close()
	for i, c := range cls {
		log.Println(i, c)
		imgDir := path.Join(root, c, "images")
		_, err := os.Stat(imgDir)
		op.CheckE(err)
		files, err := os.ReadDir(imgDir)
		op.CheckE(err)
		for _, f := range files {
			path := path.Join(imgDir, f.Name())
			file.WriteString(path + "\n")
		}

	}
	return nil
}

func main() {
	listCmd := flag.NewFlagSet("list", flag.ExitOnError)
	listDir := listCmd.String("dir", "", "Directory")
	listCls := listCmd.String("cls", "", "Classes")

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
		op.CheckE(err)
	case "crop":
		cropCmd.Parse(os.Args[2:])
		classes := strings.Split(*cropCls, ",")
		err := runCrop(*cropDir, *cropFreq, classes)
		op.CheckE(err)
	default:
		log.Println("Expected subcommands")
		os.Exit(1)
	}
}
