#!/bin/bash

DEST=/work02/SLUD2021/KB/pageviews
mkdir -p $DEST

for (( year=2015; year<=2021; year++ )) ; do
  for (( month=1; month<=12; month++ )) ; do
    [ $month -lt 10 ] && _month="0${month}" || _month=$month
    for day in "01" "15" ; do
      for hour in "00" "06" "12" "18" ; do
        fname=pageviews-${year}${_month}${day}-${hour}0000.gz
        if [ ! -f ${DEST}/ja_${fname} ] ; then
          wget -nc https://dumps.wikimedia.org/other/pageviews/${year}/${year}-${_month}/${fname} -P $DEST \
          && zcat $DEST/$fname | awk '$1 == "ja"{print}' > $DEST/${fname%%.*} \
          && rm $DEST/${fname} \
          && gzip $DEST/${fname%%.*} -c > $DEST/ja_${fname} \
          && rm $DEST/${fname%%.*}
        fi
      done
    done
  done
done