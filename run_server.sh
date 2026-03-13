# bundle exec jekyll liveserve

bundle _2.2.19_ exec jekyll build
cd _site
python3 -m http.server 4000