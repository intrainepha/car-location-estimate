package main

import (
	"flag"
	"fmt"
	"image"
	"log"
	"math/rand"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	file "github.com/intrainepha/car-location-estimation/tool/src/file"
	op "github.com/intrainepha/car-location-estimation/tool/src/ops"
	tp "github.com/intrainepha/car-location-estimation/tool/src/types"
)

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
			kt := tp.NewKITTI(l)
			clsID, err := kt.Check(cls)
			if err != nil {
				continue
			}
			bRct := tp.NewRect(kt.Rct.Xtl, kt.Rct.Ytl, kt.Rct.Xbr, kt.Rct.Ybr)
			bb := tp.NewBox(bRct, imgSize)
			offX, offY := bb.Sz.W*0.25, bb.Sz.H*0.25
			rRct := tp.NewRect(
				bb.Rct.Xtl-offX, bb.Rct.Ytl-offY,
				bb.Rct.Xbr+offX, bb.Rct.Ybr+offY,
			)
			rb := tp.NewBox(rRct, imgSize)
			// Calculate bbox relative to ROI
			oRct := tp.NewRect(
				bb.Rct.Xtl-rb.Rct.Xtl, bb.Rct.Ytl-rb.Rct.Ytl,
				bb.Rct.Xbr-rb.Rct.Xtl, bb.Rct.Ybr-rb.Rct.Ytl,
			)
			ob := tp.NewBox(oRct, &rb.Sz)
			// Write ROI image
			p := path.Join(imgDirROI, strings.Replace(f.Name(), ".png", "_"+strconv.Itoa(j)+".jpg", 1))
			subImg := file.CropImg(
				imgData, image.Rect(int(rb.Rct.Xtl), int(rb.Rct.Ytl), int(rb.Rct.Xbr), int(rb.Rct.Ybr)),
			)
			file.SaveImg(p, subImg)
			// Write ROI label txt file
			p = path.Join(labelDirROI, strings.Replace(f.Name(), ".png", "_"+strconv.Itoa(j)+".txt", 1))
			file.SaveTXT(p, clsID, ob, &kt.Loc, rb)
			// Augment
			orient := [4][4]float64{
				{0, 1, 0, 1},   // move up
				{0, -1, 0, -1}, // move down
				{-1, 0, -1, 0}, // move left
				{1, 0, 1, 0},   // move right
			}
			for k, m := range orient {
				rd := (400 + float64(rand.Intn(500))) / 1000 //random number in [0.4, 0.9]
				fmt.Println(i, m, rd)
				step := [4]float64{
					m[0] * offX * rd, m[1] * offY * rd,
					m[2] * offX * rd, m[3] * offY * rd,
				}
				bRct = tp.NewRect(
					bb.Rct.Xtl+step[0],
					bb.Rct.Ytl+step[1],
					bb.Rct.Xbr+step[2],
					bb.Rct.Ybr+step[3],
				)
				bb = tp.NewBox(bRct, tp.NewSize(imgSize.W, imgSize.H))
				rRct = tp.NewRect(
					bb.Rct.Xtl-offX, bb.Rct.Ytl-offY,
					bb.Rct.Xbr+offX, bb.Rct.Ybr+offY,
				)
				rb = tp.NewBox(rRct, imgSize)
				oRct = tp.NewRect(
					bb.Rct.Xtl-rb.Rct.Xtl, bb.Rct.Ytl-rb.Rct.Ytl,
					bb.Rct.Xbr-rb.Rct.Xtl, bb.Rct.Ybr-rb.Rct.Ytl,
				)
				ob = tp.NewBox(oRct, &rb.Sz)
				p = path.Join(imgDirROI, strings.Replace(f.Name(), ".png", "_"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+".jpg", 1))
				subImg = file.CropImg(
					imgData, image.Rect(int(rb.Rct.Xtl), int(rb.Rct.Ytl), int(rb.Rct.Xbr), int(rb.Rct.Ybr)),
				)
				file.SaveImg(p, subImg)
				// Write ROI label txt file
				p = path.Join(labelDirROI, strings.Replace(f.Name(), ".png", "_"+strconv.Itoa(j)+"_"+".txt", 1))
				file.SaveTXT(p, clsID, ob, &kt.Loc, rb)
			}
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
