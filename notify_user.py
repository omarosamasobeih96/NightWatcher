from pusher_push_notifications import PushNotifications

def notify(name, progress):
    
    try:
        beams_client = PushNotifications(
            instance_id='b2420ba3-e9d2-40f9-b45d-ee56fb2fdd10',
            secret_key='1BBD63B6843395A7E41033C50DC30740BC7E2DB2152E09B03E463F489D397F82',
        )

        response = beams_client.publish_to_interests(
            interests=['hello'],
            publish_body={
                'apns': {
                    'aps': {
                        'alert': 'New Notification'
                    }
                },
                'fcm': {
                    'notification': {
                        'title': name,
                        'body': progress
                    }
                }
            }
        )
        print(response['publishId'])
    except:
        print("Android is not working")
        
    

# notify( "Omar Sobeih",  "I love my friend Mohamed")