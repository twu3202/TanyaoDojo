#!/bin/bash
printf "%s\n" 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 | xargs -P 4 -I{} /root/better_mortal/scripts/dl_one_year.sh {}
echo "==== parallel all done $(date) ====" >> /root/autodl-tmp/data/download.log
