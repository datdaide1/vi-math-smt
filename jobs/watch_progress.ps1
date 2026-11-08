# Live progress view for every running workstream + local machine load. Read-only. Ctrl-C to stop.
$ErrorActionPreference = 'SilentlyContinue'
$env:PYTHONIOENCODING = 'utf-8'
$root = 'E:\MATH-REASONING-THESIS\vi-math-smt'
$exec = "$root\docs\research_2026-11\exec"
$kaggle = 'E:\venv_tools\Scripts\kaggle.exe'
$i = 0
$kaggleLines = @('  (checking...)')

function Last-Activity($paths) {
    $latest = $null; $n = 0
    foreach ($p in $paths) {
        $files = Get-ChildItem $p -Recurse -File | Where-Object { $_.FullName -notmatch '__pycache__' }
        foreach ($f in $files) { $n++; if (-not $latest -or $f.LastWriteTime -gt $latest) { $latest = $f.LastWriteTime } }
    }
    if ($latest) { $age = [int]((Get-Date) - $latest).TotalMinutes; return ("{0} files, last write {1} ({2} min ago)" -f $n, $latest.ToString('HH:mm:ss'), $age) }
    return '(no files yet)'
}

while ($true) {
    $out = New-Object System.Collections.Generic.List[string]
    $out.Add(("==== Progress  {0} ====" -f (Get-Date -Format 'HH:mm:ss')))

    # ---- local machine ----
    $gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null
    $os = Get-CimInstance Win32_OperatingSystem
    $cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
    $freeGB = [math]::Round($os.FreePhysicalMemory / 1MB, 1); $totGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
    if ($gpu) { $g = $gpu.Split(','); $out.Add(("[LOCAL] GPU util {0}%  VRAM {1}/{2} MiB  {3}C | CPU {4}% | RAM free {5}/{6} GB" -f $g[0].Trim(), $g[1].Trim(), $g[2].Trim(), $g[3].Trim(), $cpu, $freeGB, $totGB)) }
    else { $out.Add(("[LOCAL] GPU n/a | CPU {0}% | RAM free {1}/{2} GB" -f $cpu, $freeGB, $totGB)) }

    # ---- G1 generator pilot ----
    $out.Add('')
    $out.Add('[G1] generator pilot - API calls per model')
    $files = Get-ChildItem "$root\logs\api\G1" -Filter *.jsonl -Recurse
    if ($files) {
        $counts = @{}; $total = 0; $e429 = 0
        foreach ($f in $files) { foreach ($line in [System.IO.File]::ReadLines($f.FullName)) {
            $total++
            $m = [regex]::Match($line, '"model"\s*:\s*"([^"]+)"'); $k = if ($m.Success) { $m.Groups[1].Value } else { '?' }
            if ($counts.ContainsKey($k)) { $counts[$k]++ } else { $counts[$k] = 1 }
            if ($line -match '429') { $e429++ } } }
        foreach ($k in ($counts.Keys | Sort-Object)) { $out.Add(("  {0,-38} {1,6}" -f $k, $counts[$k])) }
        $out.Add(("  total {0}  lines-with-429 {1}" -f $total, $e429))
    }
    if (Test-Path "$exec\G1_generator_pilot.md") { $out.Add('  REPORT READY') }

    # ---- S2 Kaggle (poll ~every 2 min) ----
    $out.Add('')
    $out.Add('[S2] student pilot on Kaggle')
    if ($i % 6 -eq 0) {
        $kaggleLines = @()
        foreach ($d in (Get-ChildItem "$root\jobs\kernels" -Directory -Filter 's2-*')) {
            $meta = Get-Content "$($d.FullName)\kernel-metadata.json" -Raw | ConvertFrom-Json
            if ($meta.id) { $kaggleLines += ('  ' + (& $kaggle kernels status $meta.id 2>&1 | Select-Object -Last 1)) }
        }
        $q = & $kaggle quota 2>&1 | Select-String 'GPU'
        if ($q) { $kaggleLines += ('  quota: ' + ($q.ToString() -replace '\s+', ' ')) }
    }
    foreach ($l in $kaggleLines) { $out.Add($l) }
    if (Test-Path "$exec\S2_student_pilot.md") { $out.Add('  REPORT READY') }

    # ---- X1..X5 ----
    $out.Add('')
    $sp = 'C:\Users\tranh\AppData\Local\Temp\claude\b8848eb08e2f626f\884a7b4b-700c-4c10-ad97-d56477e30ce1\scratchpad'
    $ws = [ordered]@{
        'X1 grader'      = @("$root\vi_math_verified_aug\eval\math_verify.py", "$root\tests\test_math_verify.py", "$root\tests\grader_cases", "$sp\x1");
        'X2 benchmark'   = @("$root\vi_math_verified_aug\vi_exam", "$root\data\vi_exam", "$sp\x2");
        'X3 eval sets'   = @("$root\vi_math_verified_aug\eval_sets", "$root\data\eval", "$sp\x3");
        'X4 symbolic'    = @("$root\vi_math_verified_aug\symbolic", "$sp\x4");
        'X5 decontam'    = @("$root\vi_math_verified_aug\eval\decontam.py", "$root\data\contam", "$sp\x5");
        'Y1 train/eval'  = @("$root\vi_math_verified_aug\train", "$root\vi_math_verified_aug\evalrun", "$sp\y1");
        'Y2 analysis'    = @("$root\vi_math_verified_aug\analysis", "$sp\y2");
        'Y3 common'      = @("$root\vi_math_verified_aug\common", "$root\prompts", "$sp\y3")
    }
    $rep = @{ 'X1 grader'='X1_*.md'; 'X2 benchmark'='X2_*.md'; 'X3 eval sets'='X3_*.md'; 'X4 symbolic'='X4_*.md'; 'X5 decontam'='X5_*.md'; 'Y1 train/eval'='Y1_*.md'; 'Y2 analysis'='Y2_*.md'; 'Y3 common'='Y3_*.md' }
    foreach ($k in $ws.Keys) {
        $done = if (Get-ChildItem $exec -Filter $rep[$k]) { '  [REPORT READY]' } else { '' }
        $out.Add(("[{0}] {1}{2}" -f $k, (Last-Activity $ws[$k]), $done))
    }

    Clear-Host
    $out | ForEach-Object { Write-Host $_ }
    $i++
    Start-Sleep -Seconds 20
}
