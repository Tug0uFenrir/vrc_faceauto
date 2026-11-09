# Rebuild the static README figures from the Blender renders (Windows PowerShell).
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing
$inputDir = Join-Path $PSScriptRoot "output"
$images = Join-Path (Split-Path $PSScriptRoot) "docs/images"
[IO.Directory]::CreateDirectory($images) | Out-Null
$ink = [Drawing.Color]::FromArgb(237,233,225)
$muted = [Drawing.Color]::FromArgb(190,184,174)
function Canvas([int]$w,[int]$h) {
    $script:canvas = [Drawing.Bitmap]::new($w,$h)
    $script:graphics = [Drawing.Graphics]::FromImage($canvas)
    $graphics.PageUnit = [Drawing.GraphicsUnit]::Pixel
    $graphics.SmoothingMode = [Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.InterpolationMode = [Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $graphics.TextRenderingHint = [Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $graphics.Clear([Drawing.Color]::FromArgb(34,32,30))
}
function Caption($label,$x,$y,$size,$color) {
    $font = [Drawing.Font]::new("Microsoft YaHei UI",$size,[Drawing.FontStyle]::Regular,[Drawing.GraphicsUnit]::Pixel)
    $brush = [Drawing.SolidBrush]::new($color)
    $graphics.DrawString($label,$font,$brush,[float]$x,[float]$y)
    $brush.Dispose(); $font.Dispose()
}
function Frame($name,$x,$y,$w,$h,$source) {
    $pic = [Drawing.Bitmap]::new((Join-Path $inputDir "$name.png"))
    $graphics.DrawImage($pic,[Drawing.Rectangle]::new($x,$y,$w,$h),$source[0],$source[1],$source[2],$source[3],[Drawing.GraphicsUnit]::Pixel)
    $pic.Dispose()
}
function SaveFigure($name) {
    $canvas.Save((Join-Path $images "$name.png"),[Drawing.Imaging.ImageFormat]::Png)
    $graphics.Dispose(); $canvas.Dispose()
}
$names = @("pp","ff","aa","ih","oh","ou")
$labels = @("双唇合拢","下唇内收","张开","横向展开","圆唇","收口前突")
Canvas 1136 528
for ($i=0;$i -lt 6;$i++) {
    $x=12+($i%3)*375; $y=10+[math]::Floor($i/3)*254
    Frame $names[$i] $x $y 362 200 @(30,126,420,232)
    Caption ($names[$i].ToUpper()) ($x+3) ($y+203) 24 $ink
    Caption $labels[$i] ($x+60) ($y+208) 17 $muted
}
SaveFigure "six-visemes"
Canvas 1136 265
foreach($i in 0..2) {
    $x=12+375*$i
    Frame @("ih","ih_ou_half","ou")[$i] $x 10 362 200 @(30,126,420,232)
    Caption @("IH  100%","IH  50%   +   OU  50%","OU  100%")[$i] ($x+3) 216 20 $ink
}
SaveFigure "ih-to-ou"
Canvas 1136 350
Frame "ada" 12 10 300 300 @(0,0,480,480)
Frame "neutral" 324 10 394 300 @(43,85,394,300)
Frame "neutral" 730 10 394 300 @(43,85,394,300)
$marks=Get-Content (Join-Path $inputDir "markers.json") -Raw | ConvertFrom-Json
function Mark($x,$y,$color,$r) {
    $brush=[Drawing.SolidBrush]::new([Drawing.ColorTranslator]::FromHtml($color))
    $graphics.FillEllipse($brush,[single]($x-$r),[single]($y-$r),[single](2*$r),[single](2*$r))
    $brush.Dispose()
}
for ($i=0;$i -lt 2;$i++) {
    $color=@("#61DCE3","#FFBE73")[$i]
    $pt=$marks.seeds[$i]; Mark ($pt[0]+281) ($pt[1]-75) $color 6
    $pts=[Drawing.PointF[]]@($marks.arcs[$i]|ForEach-Object {[Drawing.PointF]::new([single]($_[0]+687),[single]($_[1]-75))})
    $pen=[Drawing.Pen]::new([Drawing.ColorTranslator]::FromHtml($color),3)
    $graphics.DrawLines($pen,$pts); $pen.Dispose()
}
foreach ($pt in $marks.corners) { Mark ($pt[0]+687) ($pt[1]-75) "#F7F6EF" 6 }
Caption "Ada · 中性素材" 16 313 18 $ink
Caption "上、下唇内缘各点一次" 328 313 18 $ink
Caption "两条追踪路径，白点为嘴角" 734 313 18 $ink
SaveFigure "lip-landmarks"
