import { createClerkClient } from '@clerk/backend'

type Args = {
    email?: string
    first?: string
    last?: string
}

function parseArgs(): Args {
    const args = process.argv.slice(2)
    const out: Args = {}
    for (let i = 0; i < args.length; i++) {
        const key = args[i]
        const val = args[i + 1]
        if (!val || val.startsWith('--')) continue
        switch (key) {
            case '--email':
                out.email = val
                break
            case '--first':
                out.first = val
                break
            case '--last':
                out.last = val
                break
            default:
                break
        }
    }
    return out
}

async function main() {
    const { email, first, last } = parseArgs()
    const secretKey = process.env.CLERK_SECRET_KEY

    if (!secretKey) {
        console.error('Missing CLERK_SECRET_KEY env var')
        process.exit(1)
    }
    if (!email) {
        console.error('Usage: --email user@example.com [--first First] [--last Last]')
        process.exit(1)
    }

    const clerk = createClerkClient({ secretKey })

    const data = await clerk.users
        .createUser({
            emailAddress: [email],
            firstName: first,
            lastName: last,
            skipPasswordRequirement: true
        })
        .catch((err) => {
            // Surface Clerk API errors with more detail
            console.error('Failed to create user')
            const anyErr = err as { errors?: unknown }
            if (anyErr?.errors) {
                console.error(JSON.stringify(anyErr.errors, null, 2))
            } else {
                console.error(err)
            }
            process.exit(1)
        })

    console.log('User created (Clerk SDK):')
    console.log(JSON.stringify(data, null, 2))
}

main().catch((err) => {
    console.error(err)
    process.exit(1)
})
