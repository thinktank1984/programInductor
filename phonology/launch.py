from datetime import datetime
import time
import subprocess32 as subprocess
import os
import json
import sys

ZONE = None

def user():
    import getpass
    return getpass.getuser()


def branch():
    return subprocess.check_output(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode("utf-8").strip()

def launchGoogleCloud(size, name):
    name = name.replace('_','-').replace('.','-').lower()
    os.system("gcloud compute --project tenenbaumlab disks create %s --size 30 --zone us-east1-b --source-snapshot phonologyoctober28 --type pd-standard"%name)
    output = \
        subprocess.check_output(["/bin/bash", "-c",
                                 "gcloud compute --project=tenenbaumlab instances create %s --zone=us-east1-b --machine-type=%s --subnet=default --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=150557817012-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --disk=name=%s,device-name=%s,mode=rw,boot=yes,auto-delete=yes"%(name,size, name, name)])
    global ZONE
    ZONE = output.decode("utf-8").split("\n")[1].split()[1]
    print "Launching in zone %s"%ZONE
    return name, name

def scp(address, localFile, remoteFile):
    global ZONE
    if arguments.google:
        command = "gcloud compute scp --zone=%s %s %s:%s"%(ZONE,localFile, address, remoteFile)
    else:
        assert False
        #command = f"scp -o StrictHostKeyChecking=no -i ~/.ssh/testing.pem {localFile} ubuntu@{address}:{remoteFile}"
    print(command)
    os.system(command)

def ssh(address, command, pipeIn=None):
    global ZONE
    if arguments.google:
        command = "gcloud compute ssh --zone=%s %s --command='%s'"%(ZONE,address, command)
    else:
        assert False
        #command = f"ssh -o StrictHostKeyChecking=no -i ~/.ssh/testing.pem ubuntu@{address} '{command}'"
    if pipeIn:
        command = "%s | %s"%(pipeIn,command)
    print(command)
    os.system(command)

def sendCheckpoint(address, checkpoint):
    print("Sending checkpoint:")
    scp(address, checkpoint, "~/%s"%(os.path.split(checkpoint)[1]))


def sendCommand(
        address,
        script,
        job_id,
        upload,
        ssh_key,
        resume,
        tar,
        shutdown,
        checkpoint=None):
    import tempfile

    br = branch()

    if checkpoint is None:
        copyCheckpoint = ""
    else:
        if '/' in checkpoint:
            checkpoint = os.path.split(checkpoint)[1]
        copyCheckpoint = "mv ~/%s ~/programInductor/phonology" % checkpoint

    preamble = """#!/bin/bash
python programInductor/phonology/command_server.py KILL
export PATH="$PATH:/home/ellisk/sketch-1.7.5/sketch-frontend"
export SKETCH_HOME="/home/ellisk/sketch-1.7.5/sketch-frontend/runtime"
export PYTHONIOENCODING=utf8
cd ~/programInductor/
%s
git fetch
git checkout %s
git pull
sudo apt-get install  -y pypy
"""%(copyCheckpoint,br)
    preamble += "mv ~/patch ~/programInductor/patch\n"
    preamble += "git apply patch \n"

    if upload:
        # This is probably a terribly insecure idea...
        # But I'm not sure what the right way of doing it is
        # I'm just going to copy over the local SSH identity
        # Assuming that this is an authorized key at the upload site this will work
        # Of course it also means that if anyone were to pull the keys off of AWS,
        # they would then have access to every machine that you have access to
        UPLOADFREQUENCY = 60 * 3  # every 3 minutes
        uploadCommand = """\
rsync  -e 'ssh  -o StrictHostKeyChecking=no' -avz \
phonology/jobs phonology/experimentOutputs {}""".format(upload)
        preamble += """
mv ~/.ssh/%s ~/.ssh/id_rsa
mv ~/.ssh/%s.pub ~/.ssh/id_rsa.pub
chmod 600 ~/.ssh/id_rsa
chmod 600 ~/.ssh/id_rsa.pub
bash -c "while sleep %d; do %s; done" &> /tmp/test.txt & 
UPLOADPID=$!
""" % (ssh_key, ssh_key, UPLOADFREQUENCY, uploadCommand)

    script = preamble + script

    if upload:
        script += """
kill -9 $UPLOADPID
cd ..
%s
""" % (uploadCommand)
    if shutdown:
        script += """
sudo shutdown -h now
"""

    fd = tempfile.NamedTemporaryFile(mode='w', delete=False, dir="/tmp")
    fd.write(script)
    fd.close()
    name = fd.name

    print("SCRIPT:")
    print(script)

    # Copy over the script
    print "Copying script over to", address
    scp(address, name, "~/script.sh")

    # delete local copy
    os.system("rm %s" % name)

    # Send keys
    if upload:
        print("Uploading your ssh identity")
        scp(address, "~/.ssh/%s"%ssh_key, "~/.ssh/%s"%(ssh_key))
        scp(address, "~/.ssh/%s.pub"%ssh_key, "~/.ssh/%s.pub"%ssh_key)

    # Send git patch
    print "Sending git patch over to", address
    os.system("git diff --stat")
    ssh(address, "cat > ~/patch",
        pipeIn="""(echo "Base-Ref: $(git rev-parse origin/%s)" ; echo ; git diff --binary origin/%s)"""%(br,br))

    # Execute the script
    # For some reason you need to pipe the output to /dev/null in order to get
    # it to detach
    ssh(address, "bash ./script.sh > /dev/null 2>&1 &")
    print("Executing script on remote host.")


def launchExperiment(
        name,
        command,
        checkpoint=None,
        tail=False,
        resume="",
        upload=None,
        ssh_key="id_rsa",
        tar=False,
        shutdown=True,
        size="t2.micro"):
    job_id = "{}_{}_{}".format(name, user(), datetime.now().strftime("%FT%T"))
    job_id = job_id.replace(":", ".")
    if upload is None and shutdown:
        print("WARNING: You didn't specify an upload host, and also specify that the machine should shut down afterwards.")

    if resume and "resume" not in command:
        print("You said to resume, but didn't give --resume to your python command. I am assuming this is a mistake.")
        sys.exit(1)
    if resume and not resume.endswith(".tar.gz"):
        print("Invalid tarball for resume.")
        sys.exit(1)

    command = "cd phonology&&%s"%command
    script = """
%s > jobs/%s 2>&1
""" % (command, job_id)

    if arguments.google:
        name = job_id
        instance, address = launchGoogleCloud(size, name)
    else:
        assert False
    time.sleep(120)
    if checkpoint is not None:
        sendCheckpoint(address, checkpoint)
    sendCommand(
        address,
        script,
        job_id,
        upload,
        ssh_key,
        resume,
        tar,
        shutdown,
        checkpoint=checkpoint)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-u', "--upload",
                        default={
                            "ellisk": "ellisk@openmind7.mit.edu:/om2/user/ellisk/phonology",
                        }.get(user(), None))
    parser.add_argument('-z', "--size",
                        default="t2.micro")
    parser.add_argument("--tail",
                        default=False,
                        help="attach to the machine and tail ec's output.",
                        action="store_true")
    parser.add_argument("--resume", metavar="TAR", type=str,
                        help="send tarball to resume checkpoint.")
    parser.add_argument("--checkpoint", metavar="checkpoint", type=str,
                        help="Send checkpoint file to resume checkpoint.")
    parser.add_argument('-k', "--shutdown",
                        default=False,
                        action="store_true")
    parser.add_argument('-c', "--google",
                        default=True,
                        action="store_true")
    parser.add_argument('-g', "--gpuImage", default=False, action='store_true')
    parser.add_argument(
        '-t',
        "--tar",
        default=False,
        help="if uploading, this sends a single tarball with relevant outputs.",
        action="store_true")
    parser.add_argument("--ssh_key", default='id_rsa', help="Name of local RSA key file for openmind.")
    parser.add_argument("name")
    parser.add_argument("command")
    arguments = parser.parse_args()

    launchExperiment(arguments.name,
                     arguments.command,
                     shutdown=arguments.shutdown,
                     tail=arguments.tail,
                     resume=arguments.resume,
                     size=arguments.size,
                     upload=arguments.upload if arguments.upload != "None" else None,
                     ssh_key=arguments.ssh_key,
                     tar=arguments.tar,
                     checkpoint=arguments.checkpoint)

"""
BILLING: https://console.aws.amazon.com/billing/home?#/
Do not go over $10k
t2.nano	1	Variable	0.5	EBS Only	$0.0058 per Hour
t2.micro	1	Variable	1	EBS Only	$0.0116 per Hour
t2.small	1	Variable	2	EBS Only	$0.023 per Hour
t2.medium	2	Variable	4	EBS Only	$0.0464 per Hour
t2.large	2	Variable	8	EBS Only	$0.0928 per Hour
t2.xlarge	4	Variable	16	EBS Only	$0.1856 per Hour
t2.2xlarge	8	Variable	32	EBS Only	$0.3712 per Hour
m4.large	2	6.5	8	EBS Only	$0.1 per Hour
m4.xlarge	4	13	16	EBS Only	$0.2 per Hour
m4.2xlarge	8	26	32	EBS Only	$0.4 per Hour
m4.4xlarge	16	53.5	64	EBS Only	$0.8 per Hour
m4.10xlarge	40	124.5	160	EBS Only	$2 per Hour
m4.16xlarge	64	188	256	EBS Only	$3.2 per Hour
Compute Optimized - Current Generation
c4.large	2	8	3.75	EBS Only	$0.1 per Hour
c4.xlarge	4	16	7.5	EBS Only	$0.199 per Hour
c4.2xlarge	8	31	15	EBS Only	$0.398 per Hour
c4.4xlarge	16	62	30	EBS Only	$0.796 per Hour
c4.8xlarge	36	132	60	EBS Only	$1.591 per Hour
GPU Instances - Current Generation
p2.xlarge	4	12	61	EBS Only	$0.9 per Hour
p2.8xlarge	32	94	488	EBS Only	$7.2 per Hour
p2.16xlarge	64	188	732	EBS Only	$14.4 per Hour
p3.2xlarge	8	23.5	61	EBS Only	$3.06 per Hour
p3.8xlarge	32	94	244	EBS Only	$12.24 per Hour
p3.16xlarge	64	188	488	EBS Only	$24.48 per Hour
g3.4xlarge	16	47	122	EBS Only	$1.14 per Hour
g3.8xlarge	32	94	244	EBS Only	$2.28 per Hour
g3.16xlarge	64	188	488	EBS Only	$4.56 per Hour
Memory Optimized - Current Generation
x1.16xlarge	64	174.5	976	1 x 1920 SSD	$6.669 per Hour
x1.32xlarge	128	349	1952	2 x 1920 SSD	$13.338 per Hour
r3.large	2	6.5	15	1 x 32 SSD	$0.166 per Hour
r3.xlarge	4	13	30.5	1 x 80 SSD	$0.333 per Hour
r3.2xlarge	8	26	61	1 x 160 SSD	$0.665 per Hour
r3.4xlarge	16	52	122	1 x 320 SSD	$1.33 per Hour
r3.8xlarge	32	104	244	2 x 320 SSD	$2.66 per Hour
r4.large	2	7	15.25	EBS Only	$0.133 per Hour
r4.xlarge	4	13.5	30.5	EBS Only	$0.266 per Hour
r4.2xlarge	8	27	61	EBS Only	$0.532 per Hour
r4.4xlarge	16	53	122	EBS Only	$1.064 per Hour
r4.8xlarge	32	99	244	EBS Only	$2.128 per Hour
r4.16xlarge	64	195	488	EBS Only	$4.256 per Hour
Storage Optimized - Current Generation
i3.large	2	7	15.25	1 x 475 NVMe SSD	$0.156 per Hour
i3.xlarge	4	13	30.5	1 x 950 NVMe SSD	$0.312 per Hour
i3.2xlarge	8	27	61	1 x 1900 NVMe SSD	$0.624 per Hour
i3.4xlarge	16	53	122	2 x 1900 NVMe SSD	$1.248 per Hour
i3.8xlarge	32	99	244	4 x 1900 NVMe SSD	$2.496 per Hour
i3.16xlarge	64	200	488	8 x 1900 NVMe SSD	$4.992 per Hour
h1.2xlarge	8	26	32	1 x 2000 HDD	$0.55 per Hour
h1.4xlarge	16	53.5	64	2 x 2000 HDD	$1.1 per Hour
h1.8xlarge	32	99	128	4 x 2000 HDD	$2.2 per Hour
h1.16xlarge	64	188	256	8 x 2000 HDD	$4.4 per Hour
d2.xlarge	4	14	30.5	3 x 2000 HDD	$0.69 per Hour
d2.2xlarge	8	28	61	6 x 2000 HDD	$1.38 per Hour
d2.4xlarge	16	56	122	12 x 2000 HDD	$2.76 per Hour
d2.8xlarge	36	116	244	24 x 2000 HDD	$5.52 per Hour
p3.16xlarge
"""
