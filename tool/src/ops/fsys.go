package ops

import (
	"log"
	"os"
)

func CheckDir(d string) bool {
	_, err := os.Stat(d)
	return os.IsNotExist(err)
}

func CleanDir(ds ...string) {
	for _, d := range ds {
		err := os.RemoveAll(d)
		if err != nil {
			log.Println(err)
		}
		err = os.MkdirAll(d, 0755)
		if err != nil {
			log.Println(err)
		}
	}
}
