# # if PowerShell scripts don't work, make sure to:
# # Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
# # in a powershell running in administrator mode.
# #
# # Sadly, there is no good equivilant to "$@" in ps1
# param($p1)

# $ip_address = (Test-Connection -ComputerName $env:COMPUTERNAME -Count 1).IPAddressToString)

# docker build --tag learning_machines . --build-arg IP_ADRESS=$ip_address
# # Mounting to a directory that does not exist creates it.
# # Mounting to relative paths works since docker engine 23
# docker run -t --rm -p 45100:45100 -p 45101:45101 -v ./results:/root/results learning_machines $PSBoundParameters["p1"]

param([string]$mode)

if ($mode -eq "--simulation") {
    Write-Host "Running in simulation mode. Starting coppeliaSim..."
	Start-Process powershell -ArgumentList " -NoExit -Command & { .\scripts\start_coppelia_sim2.ps1 .\scenes\arena_approach.ttt }" #-WindowStyle Hidden
	}
elseif ($mode -ne "--hardware") {
    Write-Host "Invalid mode or no mode specified: $mode. Either --simulation or --hardware"
	Exit
}

# Get IP address
$ipAddress = 130.37.67.235

# Build Docker image
docker build --tag lm --build-arg IP_ADRESS=$ipAddress .
Write-Host $ipAddress

# # Create IP script
# Set-Content -Path "./catkin_ws/ip.sh" -Value "#!/bin/bash`nexport GLOBAL_IP_ADRESS="$ipAddress""


# Run Docker container
# docker run -t --rm -p 45100:45100 -p 45101:45101 -v $pwd\results:/root/results -v $pwd\catkin_ws:/root/catkin_ws lm $mode
Invoke-Expression -Command "docker run -t --rm -p 45100:45100 -p 45101:45101 -v '$(Get-Location)\results:/root/results' lm --simulation"

# Change ownership of results directory
# This assumes the equivalent of "sudo chown "$USER":"$USER" ./results -R" in PowerShell
# Get-ChildItem -Path "./results" -Recurse | ForEach-Object {
#     $_ | Get-Acl | ForEach-Object {
#         $_.SetOwner([System.Security.Principal.NTAccount]::new($env:USERNAME))
#         $_ | Set-Acl $_.Path
#     }
# }
# if ($mode -eq "--simulation") {
# 	#Stop-Process -Name "coppeliaSim" -Force
# 	}