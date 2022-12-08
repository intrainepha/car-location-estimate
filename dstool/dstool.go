package main

import (
	"encoding/xml"
	"flag"
	"io/ioutil"
	"log"
	"os"
	"path"
	"path/filepath"
	"strings"
)

type XMLData struct {
	RoiSize ROISize `xml:"size"`
	Roi     Box     `xml:"roi"`
	Obj     Object  `xml:"object"`
}

type ROISize struct {
	W int `xml:"width"`
	H int `xml:"height"`
}
type Box struct {
	Xmin float64 `xml:"xmin"`
	Ymin float64 `xml:"ymin"`
	Xmax float64 `xml:"xmax"`
	Ymax float64 `xml:"ymax"`
}
type Object struct {
	Name string `xml:"name"`
	Bbox Box    `xml:"bndbox"`
}

type TxtData struct {
	ID   int
	BoxX float64
	BoxY float64
	BoxW float64
	BoxH float64
	Y    float64
	X    float64
}

func list(root string, cls []string) error {
	/*Generate paths.txt file, which contains data paths with each
	data path in one line, this file format is required by yolo-v3.

	Args:
		root(string): Directory contains data files
			dataset_directory/
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
	if err != nil {
		log.Fatal(err)
	}
	txt := path.Join(root, "paths.txt")
	buffer, err := os.OpenFile(txt, os.O_RDWR|os.O_CREATE, 0755)
	if err != nil {
		log.Fatal(err)
		return err
	}
	defer buffer.Close()
	for i, c := range cls {
		log.Println(i, c)
		imgDir := path.Join(root, c, "images")
		_, err := os.Stat(imgDir)
		if err != nil {
			log.Fatal(err)
			return err
		}
		imgs, err := os.ReadDir(imgDir)
		if err != nil {
			log.Fatal(err)
			return err
		}
		for _, f := range imgs {
			path := path.Join(imgDir, f.Name())
			buffer.WriteString(path + "\n")
		}

	}
	return nil
}

func genTxt(root string, cls []string) error {
	/*Read *.xml lable files and generate *.txt files in yolo-v3 format.

	Args:
		root(string): Dataset directory
		cls([]string): classes you choose to generate

	Returns:
		error
	*/

	root, err := filepath.Abs(root)
	if err != nil {
		log.Fatal(err)
	}
	for id, c := range cls {
		xmlDir := path.Join(root, c, "annotations")
		// txtDir := path.Join(root, c, "labels")
		xmls, err := os.ReadDir(xmlDir)
		if err != nil {
			log.Fatal(err)
			return err
		}
		for _, x := range xmls {
			xmlPath := path.Join(xmlDir, x.Name())
			data, _ := ioutil.ReadFile(xmlPath)
			var xmlData XMLData
			_ = xml.Unmarshal([]byte(data), &xmlData)
			log.Printf(
				"classid=%d, width=%d, height=%d, xmin=%g, ymin=%g, xmax=%g, xmax=%g",
				id,
				xmlData.RoiSize.W,
				xmlData.RoiSize.H,
				xmlData.Obj.Bbox.Xmin,
				xmlData.Obj.Bbox.Ymin,
				xmlData.Obj.Bbox.Xmax,
				xmlData.Obj.Bbox.Ymax,
			)
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
	if len(os.Args) < 2 {
		log.Fatal("Expected subcommands!")
		os.Exit(1)
	}
	switch os.Args[1] {
	case "list":
		listCmd.Parse(os.Args[2:])
		classes := strings.Split(*listCls, ",")
		err := list(*listDir, classes)
		if err != nil {
			log.Fatal(err)
		}
	case "gt":
		gtCmd.Parse(os.Args[2:])
		classes := strings.Split(*gtCls, ",")
		err := genTxt(*gtDir, classes)
		if err != nil {
			log.Fatal(err)
		}
	default:
		log.Println("Expected subcommands")
		os.Exit(1)
	}
}
