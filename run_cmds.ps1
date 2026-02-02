<#
run_cmds.ps1

Usage examples (PowerShell):
  cd D:\web3quant
  .\run_cmds.ps1 -Task eval_ic
  .\run_cmds.ps1 -Task eval_trade -Expr "vol_60"
  .\run_cmds.ps1 -Task llm_offline
  .\run_cmds.ps1 -Task llm_deepseek

Notes:
- This script avoids VS Code "http://_vscodecontentref_" links; it uses real paths.
- If you're using conda, activate your env first.
#>

[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [ValidateSet(
    'eval_ic',
    'eval_trade',
    'eval_hybrid',
    'llm_offline',
    'llm_deepseek',
    'llm_deepseek_stage2',
    'train_ppo',
    'export_topk'
  )]
  [string]$Task,

  # Common args
  [string]$Panel = "alphagen_panel_BTC_USDT_1m_1m_h60.csv",
  [string]$Expr = "vol_60",

  # Output overrides
  [string]$OutJson = "",
  [string]$OutCsv = "",

  # LLM
  [int]$N = 80,
  [int]$TopK = 20,
  [string]$ExprsFile = "llm_seed_exprs.txt",

  # Training
  [int]$Timesteps = 20000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = $PSScriptRoot
$PythonDefault = "D:/Anacondasoftware/python.exe"

function Resolve-Python {
  if (Test-Path $PythonDefault) {
    return $PythonDefault
  }

  $cmd = Get-Command python.exe -ErrorAction SilentlyContinue
  if ($null -ne $cmd) {
    return $cmd.Source
  }

  throw "Python not found. Either install Python or edit run_cmds.ps1 to point to your interpreter."
}

$Py = Resolve-Python

function Join-Root([string]$p) {
  return (Join-Path $Root $p)
}

function Run([string]$CommandLine) {
  Write-Host "`n==> $CommandLine`n" -ForegroundColor Cyan
  Invoke-Expression $CommandLine
}

# Ensure we run from repo root
Set-Location $Root

# Common paths
$EvalExprPy = Join-Root "alphagen_style/scripts/evaluate_expr.py"
$LlmPy = Join-Root "alphagen_style/scripts/llm_propose_and_score.py"
$TrainPy = Join-Root "alphagen_style/scripts/train_maskable_ppo.py"
$ExportPy = Join-Root "alphagen_style/scripts/export_topk_expr.py"

$PanelPath = Join-Root $Panel

if (-not (Test-Path $PanelPath)) {
  throw "Panel not found: $PanelPath"
}

switch ($Task) {
  'eval_ic' {
    if (-not $OutJson) { $OutJson = Join-Root "single_factor_report_ic.json" }
    $cmd = "$Py `"$EvalExprPy`" --panel `"$PanelPath`" --expr `"$Expr`" --reward-mode ic --out `"$OutJson`""
    Run $cmd
  }

  'eval_trade' {
    if (-not $OutJson) { $OutJson = Join-Root "single_factor_report_trade.json" }
    $cmd = "$Py `"$EvalExprPy`" --panel `"$PanelPath`" --expr `"$Expr`" --reward-mode trade --out `"$OutJson`""
    Run $cmd
  }

  'eval_hybrid' {
    if (-not $OutJson) { $OutJson = Join-Root "single_factor_report_hybrid.json" }
    $cmd = "$Py `"$EvalExprPy`" --panel `"$PanelPath`" --expr `"$Expr`" --reward-mode hybrid --use-fold-median-ic --out `"$OutJson`""
    Run $cmd
  }

  'llm_offline' {
    if (-not $OutCsv) { $OutCsv = Join-Root "llm_topk_offline.csv" }
    if (-not $OutJson) { $OutJson = Join-Root "llm_topk_offline.json" }

    $exprs = Join-Root $ExprsFile
    if (-not (Test-Path $exprs)) {
      throw "Exprs file not found: $exprs"
    }

    $cmd = "$Py `"$LlmPy`" --panel `"$PanelPath`" --exprs-file `"$exprs`" --topk $TopK --out-csv `"$OutCsv`" --out-json `"$OutJson`""
    Run $cmd
  }

  'llm_deepseek' {
    if (-not $env:DEEPSEEK_API_KEY) {
      throw "Missing DEEPSEEK_API_KEY. Run: set DEEPSEEK_API_KEY=..."
    }
    if (-not $OutCsv) { $OutCsv = Join-Root "llm_topk_deepseek.csv" }
    if (-not $OutJson) { $OutJson = Join-Root "llm_topk_deepseek.json" }

    $cmd = "$Py `"$LlmPy`" --panel `"$PanelPath`" --deepseek --n $N --topk $TopK --out-csv `"$OutCsv`" --out-json `"$OutJson`""
    Run $cmd
  }

  'llm_deepseek_stage2' {
    if (-not $env:DEEPSEEK_API_KEY) {
      throw "Missing DEEPSEEK_API_KEY. Run: set DEEPSEEK_API_KEY=YOUR_KEY_HERE"
    }
    if (-not $OutCsv) { $OutCsv = Join-Root "llm_topk_deepseek_stage2.csv" }
    if (-not $OutJson) { $OutJson = Join-Root "llm_topk_deepseek_stage2.json" }

    $cmd = "$Py `"$LlmPy`" --panel `"$PanelPath`" --deepseek --n $N --stage2 --stage2-n $N --topk $TopK --out-csv `"$OutCsv`" --out-json `"$OutJson`""
    Run $cmd
  }

  'train_ppo' {
    # NOTE: This assumes you already have gymnasium/sb3/sb3-contrib installed in the env.
    $cmd = "$Py `"$TrainPy`" --total-timesteps $Timesteps"
    Run $cmd
  }

  'export_topk' {
    # This assumes a trained MaskablePPO model exists. Adjust args inside export script as needed.
    $cmd = "$Py `"$ExportPy`""
    Run $cmd
  }

  default {
    throw "Unknown task: $Task"
  }
}
