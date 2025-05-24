import pulumi
import pulumi_aws as aws

config = pulumi.Config()
project_name = pulumi.get_project()
stack_name = pulumi.get_stack()
region = aws.config.region

# S3
buckets = {}
for name in ["dem", "imagery", "predictions"]:
    bucket = aws.s3.Bucket(
        f"{project_name}-{name}",
        acl="private",
        tags={"Project": project_name, "Type": name},
    )
    buckets[name] = bucket

# SG
sec_group = aws.ec2.SecurityGroup(
    "ec2-sg",
    description="Allow SSH and PostgreSQL",
    ingress=[
        {
            "protocol": "tcp",
            "from_port": 22,
            "to_port": 22,
            "cidr_blocks": ["0.0.0.0/0"],
        },
        {
            "protocol": "tcp",
            "from_port": 5432,
            "to_port": 5432,
            "cidr_blocks": ["0.0.0.0/0"],
        },
    ],
    egress=[
        {"protocol": "-1", "from_port": 0, "to_port": 0, "cidr_blocks": ["0.0.0.0/0"]},
    ],
)

# IAM
assume_role = aws.iam.get_policy_document(
    statements=[
        {
            "actions": ["sts:AssumeRole"],
            "principals": [{"type": "Service", "identifiers": ["ec2.amazonaws.com"]}],
        }
    ]
)

role = aws.iam.Role("ec2-role", assume_role_policy=assume_role.json)

policy = aws.iam.RolePolicy(
    "ec2-s3-policy",
    role=role.id,
    policy=buckets["predictions"].arn.apply(
        lambda arn: f"""{{
        "Version": "2012-10-17",
        "Statement": [
            {{
                "Action": ["s3:*"],
                "Effect": "Allow",
                "Resource": ["{arn}", "{arn}/*"]
            }}
        ]
    }}"""
    ),
)

instance_profile = aws.iam.InstanceProfile("ec2-profile", role=role)

# EC2
ami = aws.ec2.get_ami(
    most_recent=True,
    owners=["099720109477"],  # Canonical (Ubuntu)
    filters=[
        {
            "name": "name",
            "values": ["ubuntu/images/hvm-ssd/ubuntu-22.04-amd64-server-*"],
        }
    ],
)

ec2 = aws.ec2.Instance(
    "worker-node",
    instance_type="t3.micro",
    ami=ami.id,
    vpc_security_group_ids=[sec_group.id],
    iam_instance_profile=instance_profile.name,
    key_name=config.require("sshKeyName"),  # must exist in your AWS account
    tags={"Project": project_name, "Stack": stack_name},
)


pulumi.export("bucket_dem", buckets["dem"].id)
pulumi.export("bucket_imagery", buckets["imagery"].id)
pulumi.export("bucket_predictions", buckets["predictions"].id)
pulumi.export("instance_public_ip", ec2.public_ip)
pulumi.export("instance_id", ec2.id)
