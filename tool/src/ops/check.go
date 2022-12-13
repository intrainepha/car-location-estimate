package ops

import "log"

func CheckE(e error) {
	if e != nil {
		log.Panic(e)
	}
}
