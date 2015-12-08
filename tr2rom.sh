tr2rom() {
	sed -e 's/[âÂ]/A/g' \
		-e 's/[çÇ]/C/g' \
		-e 's/[ğĞÐð]/G/g' \
		-e 's/[ıIİİîÎÝý]/I/g' \
		-e 's/[öÖ]/O/g' \
		-e 's/[şŞÞþ]/S/g' \
		-e 's/[üÜûÛ]/U/g'
}

[ $# -ge 1 -a -f "$1" ] && input="$1" || input="-"
cat $input | tr2rom
