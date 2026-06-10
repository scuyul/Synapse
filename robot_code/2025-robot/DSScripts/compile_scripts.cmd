@echo off

del ".compiler\DSDown.bat" 2> nul
del ".compiler\temp.~b64" 2> nul

echo //4mY2xzDQo= > ".compiler\temp.~b64"
certutil.exe -f -decode ".compiler\temp.~b64" ".compiler\DSDown.bat"
del ".compiler\temp.~b64"
copy ".compiler\DSDown.bat" /b + "DSDown.bat" /b ".compiler\DSDown.bat"

del ".compiler\DSUp.bat" 2> nul
del ".compiler\temp.~b64" 2> nul

echo //4mY2xzDQo= > ".compiler\temp.~b64"
certutil.exe -f -decode ".compiler\temp.~b64" ".compiler\DSUp.bat"
del ".compiler\temp.~b64"
copy ".compiler\DSUp.bat" /b + "DSUp.bat" /b ".compiler\DSUp.bat"