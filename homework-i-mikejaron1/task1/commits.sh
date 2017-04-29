#!/bin/sh
git init task1_repo
cd task1_repo
touch 1
git add -A
git commit -m "1"
touch 2
git add -A
git commit -m "2"
touch 3
git add -A
git commit -m "3"
touch 4
git add -A
git commit -m "4"
touch 5
git add -A
git commit -m "5"

git checkout HEAD~4
git checkout -b feature
touch 6
git add -A
git commit -m "6"
touch 7
git add -A
git commit -m "7"
touch 8
git add -A
git commit -m "8"

git checkout master
git checkout HEAD~2
git checkout -b temp_branch
git rebase --onto feature temp_branch master
git checkout temp_branch
git checkout -b debug
git branch -D temp_branch

git checkout debug
touch 9
git add 9
git commit -m "9"
git log --graph --decorate --oneline --all --graph
git checkout debug
git checkout feature -- 7
git commit --amend -m 'ammending'