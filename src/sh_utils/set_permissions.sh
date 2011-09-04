find -type f -exec chmod u=rwx {} +;
find -type f -exec chmod g=r {} +;
find -type f -exec chmod o=r {} +;
find -type d -exec chmod u=rwx {} +;
find -type d -exec chmod g=rx {} +;
find -type d -exec chmod o=rx {} +;

